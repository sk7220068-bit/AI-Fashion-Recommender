package com.fashionai.recommendation;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * Implements cosine similarity computation between clothing/outfit feature vectors.
 *
 * Cosine similarity measures the angle between two vectors, making it scale-invariant.
 * Similarity = (A · B) / (||A|| * ||B||)
 *
 * Result ranges from:
 *   -1.0  → completely opposite
 *    0.0  → orthogonal (no similarity)
 *    1.0  → identical direction (most similar)
 *
 * Used by the recommendation engine to rank outfits by visual similarity.
 */
@Slf4j
@Component
public class CosineSimilarityEngine {

    /**
     * Computes the cosine similarity between two feature vectors.
     *
     * @param vectorA first feature vector (e.g., uploaded outfit features)
     * @param vectorB second feature vector (e.g., dataset outfit features)
     * @return similarity score in range [0.0, 1.0], or 0.0 if computation fails
     */
    public double computeSimilarity(List<Double> vectorA, List<Double> vectorB) {
        if (vectorA == null || vectorB == null) {
            log.warn("Null feature vector received — returning 0.0 similarity");
            return 0.0;
        }
        if (vectorA.size() != vectorB.size()) {
            log.warn("Feature vector dimension mismatch: {} vs {} — returning 0.0",
                    vectorA.size(), vectorB.size());
            return 0.0;
        }
        if (vectorA.isEmpty()) {
            return 0.0;
        }

        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;

        for (int i = 0; i < vectorA.size(); i++) {
            double a = vectorA.get(i);
            double b = vectorB.get(i);
            dotProduct += a * b;
            normA += a * a;
            normB += b * b;
        }

        // Guard against division by zero for zero-magnitude vectors
        if (normA == 0.0 || normB == 0.0) {
            log.warn("Zero-magnitude vector detected — returning 0.0 similarity");
            return 0.0;
        }

        double similarity = dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));

        // Clamp to [0, 1] — negative cosine similarity is treated as 0 for fashion use case
        return Math.max(0.0, Math.min(1.0, similarity));
    }

    /**
     * Computes similarity using raw double arrays (performance-optimized variant).
     *
     * @param vectorA first feature vector as primitive array
     * @param vectorB second feature vector as primitive array
     * @return cosine similarity in [0.0, 1.0]
     */
    public double computeSimilarity(double[] vectorA, double[] vectorB) {
        if (vectorA == null || vectorB == null || vectorA.length != vectorB.length) {
            return 0.0;
        }

        double dotProduct = 0.0, normA = 0.0, normB = 0.0;
        for (int i = 0; i < vectorA.length; i++) {
            dotProduct += vectorA[i] * vectorB[i];
            normA += vectorA[i] * vectorA[i];
            normB += vectorB[i] * vectorB[i];
        }

        if (normA == 0.0 || normB == 0.0) return 0.0;
        return Math.max(0.0, dotProduct / (Math.sqrt(normA) * Math.sqrt(normB)));
    }

    /**
     * Computes the average (centroid) of multiple feature vectors.
     * Used to create an aggregate outfit vector from individual item vectors.
     *
     * @param vectors list of item feature vectors
     * @return averaged feature vector, or empty list if input is null/empty
     */
    public List<Double> computeCentroid(List<List<Double>> vectors) {
        if (vectors == null || vectors.isEmpty()) {
            return List.of();
        }

        int dim = vectors.get(0).size();
        double[] centroid = new double[dim];

        for (List<Double> vector : vectors) {
            if (vector.size() != dim) continue; // skip malformed vectors
            for (int i = 0; i < dim; i++) {
                centroid[i] += vector.get(i);
            }
        }

        double count = vectors.size();
        return java.util.Arrays.stream(centroid)
                .mapToObj(v -> v / count)
                .collect(java.util.stream.Collectors.toList());
    }
}
