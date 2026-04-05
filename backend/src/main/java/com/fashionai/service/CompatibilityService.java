package com.fashionai.service;

import com.fashionai.compatibility.CompatibilityScorer;
import com.fashionai.model.ClothingItem;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * Service facade over the CompatibilityScorer.
 * Computes pairwise and overall outfit compatibility scores.
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class CompatibilityService {

    private final CompatibilityScorer compatibilityScorer;

    /**
     * Scores the overall compatibility of a detected outfit.
     *
     * @param items list of clothing items in the outfit
     * @return compatibility score in [0.0, 1.0]
     */
    public double scoreOutfitCompatibility(List<ClothingItem> items) {
        if (items == null || items.isEmpty())
            return 0.0;

        double score = compatibilityScorer.scoreOutfit(items);
        log.info("Outfit compatibility score for {} items: {:.2f}", items.size(), score);
        return score;
    }

    /**
     * Scores the compatibility between exactly two clothing items.
     *
     * @param itemA first clothing item
     * @param itemB second clothing item
     * @return pair compatibility score in [0.0, 1.0]
     */
    public double scorePairCompatibility(ClothingItem itemA, ClothingItem itemB) {
        return compatibilityScorer.scorePair(itemA, itemB);
    }

    /**
     * Returns a human-readable label for a compatibility score.
     *
     * @param score compatibility score
     * @return label string: "Excellent" | "Good" | "Fair" | "Poor"
     */
    public String getCompatibilityLabel(double score) {
        if (score >= 0.85)
            return "Excellent";
        if (score >= 0.70)
            return "Good";
        if (score >= 0.50)
            return "Fair";
        return "Poor";
    }
}
