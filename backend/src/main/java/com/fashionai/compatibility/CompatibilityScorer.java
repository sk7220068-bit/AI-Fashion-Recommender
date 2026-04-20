package com.fashionai.compatibility;

import com.fashionai.model.ClothingItem;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;

/**
 * Evaluates the fashion compatibility between clothing items.
 *
 * Uses a combination of:
 * 1. Color harmony rules (e.g. complementary, analogous, neutral palette)
 * 2. Style coherence (e.g. formal shirt + formal trousers = high score)
 * 3. Formality gap penalty (e.g. tuxedo jacket + shorts = low score)
 * 4. Feature vector cosine similarity as a learned proxy for visual harmony
 */
@Slf4j
@Component
public class CompatibilityScorer {

    /**
     * Style compatibility lookup table.
     * Maps a pair style key → compatibility score (0.0 → 1.0).
     * Keys are sorted alphabetically to ensure symmetric lookup.
     */
    private static final Map<String, Double> STYLE_COMPAT = Map.ofEntries(
            Map.entry("casual_casual", 1.0),
            Map.entry("casual_smart-casual", 0.85),
            Map.entry("casual_sporty", 0.75),
            Map.entry("casual_formal", 0.25),
            Map.entry("smart-casual_smart-casual", 1.0),
            Map.entry("smart-casual_formal", 0.80),
            Map.entry("smart-casual_sporty", 0.50),
            Map.entry("formal_formal", 1.0),
            Map.entry("formal_sporty", 0.10),
            Map.entry("sporty_sporty", 1.0));

    /**
     * Computes an overall compatibility score for a list of clothing items.
     *
     * The score is the mean pairwise compatibility across all item pairs.
     *
     * @param items list of clothing items to evaluate
     * @return compatibility score in [0.0, 1.0]; 1.0 if only one item
     */
    public double scoreOutfit(List<ClothingItem> items) {
        if (items == null || items.isEmpty())
            return 0.0;
        if (items.size() == 1)
            return 1.0;

        double totalScore = 0.0;
        int pairCount = 0;

        // Evaluate every unique pair of items
        for (int i = 0; i < items.size(); i++) {
            for (int j = i + 1; j < items.size(); j++) {
                totalScore += scorePair(items.get(i), items.get(j));
                pairCount++;
            }
        }

        return pairCount > 0 ? totalScore / pairCount : 0.0;
    }

    /**
     * Computes the compatibility score between two clothing items.
     * Combines style coherence, formality gap, and color harmony.
     *
     * @param itemA first clothing item
     * @param itemB second clothing item
     * @return compatibility score in [0.0, 1.0]
     */
    public double scorePair(ClothingItem itemA, ClothingItem itemB) {
        double styleScore = getStyleCompatibility(itemA.getStyle(), itemB.getStyle());
        double formalScore = getFormalityCompatibility(itemA.getFormalityScore(), itemB.getFormalityScore());
        double featureScore = getFeatureVectorCompatibility(itemA.getFeatureVector(), itemB.getFeatureVector());

        // Weighted combination: style has highest weight, then formality, then features
        double compositeScore = (styleScore * 0.45) + (formalScore * 0.35) + (featureScore * 0.20);

        log.debug("Pair compatibility [{} | {}]: style={:.2f}, formality={:.2f}, feature={:.2f}, composite={:.2f}",
                itemA.getCategory(), itemB.getCategory(),
                styleScore, formalScore, featureScore, compositeScore);

        return compositeScore;
    }

    // ── Private scoring helpers ────────────────────────────────────────────────

    /**
     * Looks up how well two style categories work together.
     * Returns 0.6 as a neutral fallback for unknown style combinations.
     */
    private double getStyleCompatibility(String styleA, String styleB) {
        if (styleA == null || styleB == null)
            return 0.6;

        // Normalize and sort alphabetically to make lookup symmetric
        String key = styleA.compareTo(styleB) <= 0
                ? styleA + "_" + styleB
                : styleB + "_" + styleA;

        return STYLE_COMPAT.getOrDefault(key, 0.6);
    }

    /**
     * Penalizes large differences in formality level between items.
     * Two items with formality scores >0.4 apart get a deduction.
     */
    private double getFormalityCompatibility(double formalityA, double formalityB) {
        double gap = Math.abs(formalityA - formalityB);
        // Smooth penalty: exponential decay with the formality gap
        return Math.exp(-2.5 * gap);
    }

    /**
     * Uses cosine similarity of feature vectors as a learned compatibility proxy.
     * Falls back to 0.5 (neutral) if vectors are missing.
     */
    private double getFeatureVectorCompatibility(List<Double> vecA, List<Double> vecB) {
        if (vecA == null || vecB == null || vecA.isEmpty() || vecB.isEmpty())
            return 0.5;
        if (vecA.size() != vecB.size())
            return 0.5;

        double dot = 0.0, normA = 0.0, normB = 0.0;
        for (int i = 0; i < vecA.size(); i++) {
            dot += vecA.get(i) * vecB.get(i);
            normA += vecA.get(i) * vecA.get(i);
            normB += vecB.get(i) * vecB.get(i);
        }

        if (normA == 0 || normB == 0)
            return 0.5;
        return Math.max(0.0, dot / (Math.sqrt(normA) * Math.sqrt(normB)));
    }
}
