package com.fashionai.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Represents a single outfit recommendation result returned to the client.
 * Wraps an Outfit with its computed similarity score relative to the query.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class OutfitRecommendation {

    /** The recommended outfit */
    private Outfit outfit;

    /**
     * Cosine similarity score between query and this outfit's feature vector.
     * Range: 0.0 (completely different) → 1.0 (identical).
     */
    private double similarityScore;

    /**
     * Compatibility score for this outfit given the requested occasion.
     * Range: 0.0 → 1.0
     */
    private double occasionCompatibilityScore;

    /**
     * Final composite ranking score used for sorting.
     * Combines similarity and occasion compatibility.
     */
    private double rankingScore;

    /** Human-readable explanation of why this outfit was recommended */
    private String recommendationReason;

    /** Ranking position in the sorted results list */
    private int rank;
}
