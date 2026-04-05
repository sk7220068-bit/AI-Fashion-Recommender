package com.fashionai.service;

import com.fashionai.model.ClothingItem;
import com.fashionai.model.OutfitRecommendation;
import com.fashionai.model.UpgradeResult;
import com.fashionai.repository.RecommendationHistoryRepository;
import com.fashionai.upgrade.OutfitUpgradeEngine;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.util.List;

/**
 * Orchestrates the full outfit upgrade pipeline:
 *  1. Score outfit compatibility
 *  2. Apply upgrade rules via OutfitUpgradeEngine
 *  3. Fetch complementary outfit recommendations
 *  4. Optionally generate AI description
 *  5. Persist result to history
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class OutfitUpgradeService {

    private final CompatibilityService compatibilityService;
    private final OutfitUpgradeEngine upgradeEngine;
    private final RecommendationService recommendationService;
    private final RecommendationHistoryRepository historyRepository;
    private final FeatureExtractionService featureExtractionService;

    @Value("${openai.api.enabled:false}")
    private boolean openAiEnabled;

    /**
     * Full outfit upgrade pipeline.
     *
     * @param detectedItems items detected from the uploaded image
     * @param occasion      target occasion chosen by the user
     * @param imageBytes    raw image bytes (used for feature extraction)
     * @return              complete UpgradeResult with suggestions and recommendations
     */
    public UpgradeResult upgradeOutfit(List<ClothingItem> detectedItems,
                                        String occasion,
                                        byte[] imageBytes) {
        log.info("Starting outfit upgrade for occasion '{}' with {} detected items",
                occasion, detectedItems.size());

        // ── Step 1: Feature extraction ────────────────────────────────────────
        List<ClothingItem> enrichedItems = featureExtractionService
                .extractFeatures(imageBytes, detectedItems);

        // ── Step 2: Compatibility scoring ─────────────────────────────────────
        double compatScore = compatibilityService.scoreOutfitCompatibility(enrichedItems);
        log.info("Outfit compatibility score: {:.3f}", compatScore);

        // ── Step 3: Upgrade analysis ─────────────────────────────────────────
        UpgradeResult result = upgradeEngine.analyze(enrichedItems, occasion, compatScore);

        // ── Step 4: Recommendations ──────────────────────────────────────────
        List<Double> aggregateVector = featureExtractionService.computeAggregateVector(enrichedItems);
        List<OutfitRecommendation> recommendations;

        if (!aggregateVector.isEmpty()) {
            recommendations = recommendationService.recommend(aggregateVector, occasion, null);
        } else {
            recommendations = recommendationService.recommendByOccasionAndStyle(occasion, null);
        }
        result.setRecommendations(recommendations);

        // ── Step 5: Optional AI description ─────────────────────────────────
        if (openAiEnabled) {
            String aiDesc = generateAiDescription(detectedItems, occasion, result);
            result.setAiGeneratedDescription(aiDesc);
        }

        // ── Step 6: Set timestamp and persist ────────────────────────────────
        result.setCreatedAt(Instant.now());
        UpgradeResult saved = historyRepository.save(result);
        log.info("Saved upgrade result with id '{}'", saved.getId());

        return saved;
    }

    /**
     * Retrieves the 20 most recent upgrade analyses.
     */
    public List<UpgradeResult> getRecentHistory() {
        return historyRepository.findTop20ByOrderByCreatedAtDesc();
    }

    /**
     * Optional: generates a natural language outfit description using OpenAI API.
     * Currently returns a template string; wire to OpenAI client when API key is configured.
     */
    private String generateAiDescription(List<ClothingItem> items,
                                           String occasion,
                                           UpgradeResult result) {
        // Build a prompt summary — in production this would call OpenAI GPT-4o
        StringBuilder prompt = new StringBuilder("Upgrade for ")
                .append(occasion)
                .append(" occasion. Detected: ");

        items.forEach(item -> prompt.append(item.getCategory()).append(", "));

        if (!result.getItemsToReplace().isEmpty()) {
            prompt.append("Suggestions: ").append(String.join("; ", result.getUpgradeSuggestions()));
        }

        // TODO: Replace with actual OpenAI API call when key is configured
        return "AI-powered upgrade: " + result.getUpgradeSummary();
    }
}
