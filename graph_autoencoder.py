"""Learned time-conditioned patient/feature bipartite graph autoencoder."""
import numpy as np

from data_gen import cycleLength, symptomNames


N_FEATURES = len(symptomNames)
N_NODES = cycleLength * N_FEATURES


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30, 30)))


def build_patient_feature_edges(events, observation_mask, heldout=None):
    """Return observed positive patient-to-day-feature edges for message passing."""
    events = np.asarray(events, dtype=np.float64).reshape(-1, N_NODES)
    observed = np.asarray(observation_mask, dtype=bool).reshape(-1, N_NODES)
    if events.shape != observed.shape:
        raise ValueError("events and observation mask must have matching shapes")
    edge_mask = observed & (events > 0)
    if heldout is not None:
        edge_mask &= ~np.asarray(heldout, dtype=bool).reshape(-1, N_NODES)
    return edge_mask.astype(np.float64)


class BipartiteGraphAutoencoder:
    """Trainable patient and time-conditioned feature nodes with bipartite passing.

    Every feature node is a (day, category) pair, parameterized by shared day
    and category embeddings. Training nodes exchange messages over observed
    positive patient-feature edges, then a tied inner-product decoder reconstructs
    masked positives and sampled observed zeros.
    """

    def __init__(self, patient_count, embedding_dim=16, seed=0, learning_rate=0.01):
        if patient_count < 1:
            raise ValueError("patient_count must be positive")
        self.rng = np.random.default_rng(seed)
        self.learning_rate = learning_rate
        scale = 1.0 / np.sqrt(embedding_dim)
        self.patient_embeddings = self.rng.normal(
            0, scale, (patient_count, embedding_dim)
        )
        self.day_embeddings = self.rng.normal(
            0, scale, (cycleLength, embedding_dim)
        )
        self.category_embeddings = self.rng.normal(
            0, scale, (N_FEATURES, embedding_dim)
        )
        self.feature_bias = np.full(N_NODES, -2.0)
        self.patient_states = None
        self.feature_states = None

    def _base_feature_embeddings(self):
        return (
            self.day_embeddings[:, None, :] + self.category_embeddings[None, :, :]
        ).reshape(N_NODES, -1)

    @staticmethod
    def _normalized_edges(edges):
        patient_degree = np.maximum(edges.sum(axis=1), 1.0)
        feature_degree = np.maximum(edges.sum(axis=0), 1.0)
        return edges / np.sqrt(patient_degree[:, None] * feature_degree[None, :])

    def _encode_fit_graph(self, edges, patient_embeddings=None, feature_embeddings=None):
        patient_embeddings = self.patient_embeddings if patient_embeddings is None else patient_embeddings
        feature_embeddings = (
            self._base_feature_embeddings()
            if feature_embeddings is None else feature_embeddings
        )
        normalized = self._normalized_edges(edges)
        patient_states = patient_embeddings + np.dot(normalized, feature_embeddings)
        feature_states = feature_embeddings + np.dot(normalized.T, patient_embeddings)
        return patient_states, feature_states

    def fit(
        self, events, observation_mask=None, epochs=80, negative_ratio=4,
        validation_events=None, validation_mask=None, patience=8,
    ):
        events = np.asarray(events, dtype=np.float64).reshape(-1, N_NODES)
        if events.shape[0] != self.patient_embeddings.shape[0]:
            raise ValueError("event rows must match patient_count")
        observed = (
            np.ones_like(events, dtype=bool)
            if observation_mask is None
            else np.asarray(observation_mask, dtype=bool).reshape(-1, N_NODES)
        )
        if observed.shape != events.shape:
            raise ValueError("observation mask must match event matrix")
        events = np.where(observed, events, 0)

        params = [
            self.patient_embeddings,
            self.day_embeddings,
            self.category_embeddings,
            self.feature_bias,
        ]
        first = [np.zeros_like(param) for param in params]
        second = [np.zeros_like(param) for param in params]
        best_state = None
        best_validation_loss = np.inf
        stale_epochs = 0
        self.validation_losses = []
        self.epochs_trained = 0

        for epoch in range(max(1, epochs)):
            self.epochs_trained = epoch + 1
            heldout = observed & (events > 0) & (
                self.rng.random(events.shape) < 0.2
            )
            # Ensure some positive supervision even in sparse/small fixtures.
            if not heldout.any():
                positive_positions = np.argwhere(observed & (events > 0))
                if len(positive_positions):
                    chosen = positive_positions[self.rng.integers(len(positive_positions))]
                    heldout[tuple(chosen)] = True
            graph_edges = build_patient_feature_edges(events, observed, heldout)
            patient_states, feature_states = self._encode_fit_graph(graph_edges)
            self.patient_states, self.feature_states = patient_states, feature_states
            negatives = observed & (events == 0)
            negative_probability = min(
                1.0,
                negative_ratio * max(int(heldout.sum()), 1) /
                max(int(negatives.sum()), 1),
            )
            negatives &= self.rng.random(events.shape) < negative_probability
            target_mask = heldout | negatives
            if not target_mask.any():
                continue
            target = heldout.astype(np.float64)
            logits = np.dot(patient_states, feature_states.T) + self.feature_bias
            probabilities = sigmoid(logits)
            delta = (probabilities - target) * target_mask / target_mask.sum()

            grad_patient_state = np.dot(delta, feature_states)
            grad_feature_state = np.dot(delta.T, patient_states)
            normalized = self._normalized_edges(graph_edges)
            grad_patient = grad_patient_state + np.dot(normalized, grad_feature_state)
            grad_feature = grad_feature_state + np.dot(normalized.T, grad_patient_state)
            grad_day = grad_feature.reshape(cycleLength, N_FEATURES, -1).sum(axis=1)
            grad_category = grad_feature.reshape(cycleLength, N_FEATURES, -1).sum(axis=0)
            gradients = [
                grad_patient + 1e-4 * self.patient_embeddings,
                grad_day + 1e-4 * self.day_embeddings,
                grad_category + 1e-4 * self.category_embeddings,
                delta.sum(axis=0),
            ]
            gradients = [
                grad / max(1.0, np.linalg.norm(grad) / 5.0)
                for grad in gradients
            ]
            step = epoch + 1
            for index, (parameter, gradient) in enumerate(zip(params, gradients)):
                first[index] = 0.9 * first[index] + 0.1 * gradient
                second[index] = 0.999 * second[index] + 0.001 * gradient * gradient
                m_hat = first[index] / (1 - 0.9 ** step)
                v_hat = second[index] / (1 - 0.999 ** step)
                parameter -= self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)

            if validation_events is not None and validation_mask is not None:
                loss = self._validation_loss(
                    validation_events, validation_mask, events, observed
                )
                self.validation_losses.append(loss)
                if loss < best_validation_loss:
                    best_validation_loss = loss
                    best_state = [param.copy() for param in params]
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                    if stale_epochs >= patience:
                        break

        if best_state is not None:
            (self.patient_embeddings, self.day_embeddings,
             self.category_embeddings, self.feature_bias) = best_state
        self.best_validation_loss = (
            None if not np.isfinite(best_validation_loss) else float(best_validation_loss)
        )
        final_edges = build_patient_feature_edges(events, observed)
        self.patient_states, self.feature_states = self._encode_fit_graph(final_edges)
        return self

    def _validation_loss(self, validation_events, validation_mask, train_events, train_mask):
        validation_events = np.asarray(validation_events, dtype=np.float64).reshape(-1, N_NODES)
        validation_mask = np.asarray(validation_mask, dtype=bool).reshape(-1, N_NODES)
        # Validation patients are encoded only from their visible events and
        # the learned fit-patient states; their held-out values do not create edges.
        rows = []
        targets = []
        for index, (row, mask) in enumerate(zip(validation_events, validation_mask)):
            visible_edges = ((row > 0) & mask).astype(np.float64)
            query_state, _ = self._infer_patient_state(
                visible_edges, mask, train_events, train_mask
            )
            row_logits = np.dot(query_state, self.feature_states.T)[0] + self.feature_bias
            selected = mask
            if selected.any():
                rows.append(np.logaddexp(0, row_logits[selected]) - row[selected] * row_logits[selected])
                targets.append(int(selected.sum()))
        if not rows:
            return np.inf
        return float(sum(np.sum(row) for row in rows) / sum(targets))

    def _infer_patient_state(
        self, visible_edges, query_mask, reference_events, reference_mask,
        exclude_self=False,
    ):
        visible_edges = np.asarray(visible_edges, dtype=np.float64).reshape(-1, N_NODES)
        query_mask = np.asarray(query_mask, dtype=bool).reshape(-1, N_NODES)
        reference_events = np.asarray(reference_events, dtype=np.float64).reshape(-1, N_NODES)
        reference_mask = np.asarray(reference_mask, dtype=bool).reshape(-1, N_NODES)
        feature_base = self._base_feature_embeddings()
        observed_state = np.dot(visible_edges, feature_base)
        counts = np.maximum(visible_edges.sum(axis=1, keepdims=True), 1.0)
        observed_state = observed_state / np.sqrt(counts)
        similarities = np.zeros((len(visible_edges), len(reference_events)))
        for row, query in enumerate(visible_edges):
            common = query_mask[row][None, :] & reference_mask
            query_positive = (query > 0) & query_mask[row]
            reference_positive = (reference_events > 0) & reference_mask
            intersection = np.sum(common & query_positive & reference_positive, axis=1)
            query_count = np.sum(common & query_positive, axis=1)
            reference_count = np.sum(common & reference_positive, axis=1)
            denominator = np.sqrt(query_count * reference_count)
            scores = np.divide(
                intersection, denominator, out=np.zeros_like(denominator),
                where=denominator > 0,
            )
            if exclude_self and len(visible_edges) == len(reference_events):
                scores[row] = 0
            candidates = np.flatnonzero(scores > 0)
            if len(candidates):
                selected = candidates[np.argsort(scores[candidates])[-10:]]
                weights = scores[selected]
                similarities[row, selected] = weights / weights.sum()
        # Message passing from related fit patients supplies patient identity
        # information for new query nodes with sparse event neighborhoods.
        neighbor_state = np.dot(similarities, self.patient_states)
        return observed_state + neighbor_state, similarities

    def predict(
        self, observed, observation_mask, reference_events, reference_mask,
        exclude_self=False, return_neighbor_counts=False,
    ):
        observed = np.asarray(observed, dtype=np.float64).reshape(-1, N_NODES)
        mask = np.asarray(observation_mask, dtype=bool).reshape(-1, N_NODES)
        visible = np.where(mask, observed, 0)
        edges = ((visible > 0) & mask).astype(np.float64)
        patient_states, similarities = self._infer_patient_state(
            edges, mask, reference_events, reference_mask,
            exclude_self=exclude_self,
        )
        feature_states = self.feature_states
        logits = np.dot(patient_states, feature_states.T) + self.feature_bias
        scores = sigmoid(logits)
        if return_neighbor_counts:
            return scores, np.sum(similarities > 0, axis=1)
        return scores
