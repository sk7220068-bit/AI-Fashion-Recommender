package com.fashionai.upgrade;

import com.fashionai.model.ClothingItem;
import com.fashionai.model.UpgradeResult;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Rule-based Outfit Upgrade Engine.
 *
 * Analyzes a detected outfit against a target occasion and generates:
 * - Items to replace (poorly fitting for the occasion)
 * - Items to add (missing for the occasion)
 * - Specific actionable style upgrade suggestions
 * - A human-readable summary
 *
 * Example:
 * Input: T-shirt + jeans + sneakers, Occasion: party
 * Output: Replace sneakers → boots, Add blazer, Consider dark jeans or dress
 * trousers
 */
@Slf4j
@Component
public class OutfitUpgradeEngine {

        /**
         * Per-occasion upgrade rules.
         * Each entry lists items to avoid, items to prefer, and a minimum formality
         * level.
         */
        private static final Map<String, OccasionRule> OCCASION_RULES = Map.of(
                        "party", new OccasionRule(
                                        List.of("hoodie", "shorts", "flip-flops", "gym-wear", "sweatpants"),
                                        List.of("blazer", "dress", "heels", "boots", "clutch bag", "fitted trousers"),
                                        0.65),
                        "work", new OccasionRule(
                                        List.of("shorts", "flip-flops", "crop-top", "gym-wear", "oversized hoodie"),
                                        List.of("blazer", "formal trousers", "formal shirt", "heels", "oxford shoes",
                                                        "pencil skirt"),
                                        0.75),
                        "casual", new OccasionRule(
                                        List.of("tuxedo", "ballgown", "suit"),
                                        List.of("jeans", "t-shirt", "sneakers", "hoodie", "casual jacket"),
                                        0.1),
                        "date", new OccasionRule(
                                        List.of("gym-wear", "torn clothing", "flip-flops"),
                                        List.of("fitted dress", "boots", "smart casual shirt", "chinos", "loafers"),
                                        0.50),
                        "formal", new OccasionRule(
                                        List.of("jeans", "sneakers", "t-shirt", "shorts", "hoodie"),
                                        List.of("suit", "evening dress", "heels", "formal shirt", "tie",
                                                        "oxford shoes"),
                                        0.90),
                        "sport", new OccasionRule(
                                        List.of("formal suit", "heels", "dress shoes", "blazer"),
                                        List.of("tracksuit", "athletic shoes", "sports top", "leggings"),
                                        0.05));

        /** Upgrade suggestion templates for common item replacements */
        private static final Map<String, Map<String, String>> UPGRADE_SUGGESTIONS = Map.of(
                        "party", Map.of(
                                        "sneakers", "Replace sneakers with ankle boots or heels for a party look",
                                        "hoodie", "Swap the hoodie for a fitted blazer to elevate the outfit",
                                        "shorts", "Replace shorts with fitted trousers or a midi skirt",
                                        "t-shirt", "Consider a silk blouse or fitted top instead of a plain T-shirt",
                                        "flip-flops", "Swap flip-flops for strappy heels or block-heel sandals"),
                        "work", Map.of(
                                        "sneakers",
                                        "Replace sneakers with loafers or oxford shoes for a professional look",
                                        "jeans", "Consider formal trousers or chinos in place of jeans",
                                        "hoodie", "Replace the hoodie with a structured blazer",
                                        "t-shirt", "Swap the T-shirt for a formal shirt or blouse",
                                        "shorts", "Replace shorts with formal trousers or a pencil skirt"),
                        "formal", Map.of(
                                        "sneakers", "Replace sneakers with leather oxford shoes or heels",
                                        "jeans", "Replace jeans with formal dress trousers or an evening skirt",
                                        "t-shirt", "Swap the T-shirt for a formal shirt, blouse, or evening top",
                                        "hoodie", "Replace the hoodie with a tailored dinner jacket or evening wrap",
                                        "casual jacket", "Upgrade to a tuxedo jacket or formal blazer"));

        /**
         * Analyzes the detected outfit for the given occasion and
         * produces an actionable UpgradeResult with suggestions.
         *
         * @param detectedItems clothing items detected in the uploaded image
         * @param occasion      target occasion (party | work | casual | date | formal |
         *                      sport)
         * @param compatScore   pre-computed outfit compatibility score from
         *                      CompatibilityScorer
         * @return populated UpgradeResult with upgrade suggestions
         */
        public UpgradeResult analyze(List<ClothingItem> detectedItems,
                        String occasion,
                        double compatScore) {
                log.info("Analyzing outfit for occasion '{}' with {} items", occasion, detectedItems.size());

                String normalizedOccasion = occasion != null ? occasion.toLowerCase().trim() : "casual";
                OccasionRule rule = OCCASION_RULES.getOrDefault(normalizedOccasion, OCCASION_RULES.get("casual"));

                List<String> itemsToReplace = new ArrayList<>();
                List<String> itemsToAdd = new ArrayList<>();
                List<String> suggestions = new ArrayList<>();

                Set<String> detectedCategories = detectedItems.stream()
                                .map(item -> item.getCategory().toLowerCase())
                                .collect(Collectors.toSet());

                // ── Step 1: Identify items that violate occasion rules ────────────────
                for (ClothingItem item : detectedItems) {
                        String category = item.getCategory().toLowerCase();

                        if (rule.itemsToAvoid().contains(category)) {
                                itemsToReplace.add(item.getCategory());

                                // Look up a specific upgrade suggestion for this item + occasion
                                Map<String, String> occasionSuggestions = UPGRADE_SUGGESTIONS
                                                .getOrDefault(normalizedOccasion, Map.of());
                                String suggestion = occasionSuggestions.get(category);

                                if (suggestion != null) {
                                        suggestions.add(suggestion);
                                } else {
                                        suggestions.add(String.format(
                                                        "Consider replacing your %s with something more suitable for %s",
                                                        item.getCategory(), normalizedOccasion));
                                }
                        }
                }

                // ── Step 2: Identify missing preferred items ──────────────────────────
                for (String preferred : rule.preferredItems()) {
                        boolean alreadyPresent = detectedCategories.stream()
                                        .anyMatch(cat -> cat.contains(preferred.toLowerCase()) ||
                                                        preferred.toLowerCase().contains(cat));
                        if (!alreadyPresent) {
                                itemsToAdd.add(preferred);
                        }
                }

                // Cap recommendations at top 3 to keep suggestions actionable
                if (itemsToAdd.size() > 3) {
                        itemsToAdd = itemsToAdd.subList(0, 3);
                }

                // ── Step 3: Generate accessory suggestions for low-scoring outfits ─────
                if (compatScore < 0.6 && !normalizedOccasion.equals("casual")) {
                        suggestions.add("Consider a matching belt or accessory to tie the look together");
                }

                // ── Step 4: Build human-readable summary ─────────────────────────────
                String summary = buildSummary(normalizedOccasion, detectedCategories,
                                itemsToReplace, itemsToAdd, compatScore);

                return UpgradeResult.builder()
                                .occasion(occasion)
                                .detectedItems(detectedItems)
                                .overallCompatibilityScore(compatScore)
                                .itemsToReplace(itemsToReplace)
                                .upgradeSuggestions(suggestions)
                                .itemsToAdd(itemsToAdd)
                                .upgradeSummary(summary)
                                .build();
        }

        /**
         * Builds a concise human-readable upgrade summary paragraph.
         */
        private String buildSummary(String occasion,
                        Set<String> detected,
                        List<String> toReplace,
                        List<String> toAdd,
                        double score) {
                StringBuilder sb = new StringBuilder();
                sb.append(String.format("Your outfit (containing: %s) scores %.0f%% compatibility for a %s occasion. ",
                                String.join(", ", detected), score * 100, occasion));

                if (toReplace.isEmpty() && toAdd.isEmpty()) {
                        sb.append("Your outfit is well-suited for this occasion — great choice!");
                } else {
                        if (!toReplace.isEmpty()) {
                                sb.append(String.format("Consider replacing: %s. ", String.join(", ", toReplace)));
                        }
                        if (!toAdd.isEmpty()) {
                                sb.append(String.format("You could also add: %s to complete the look.",
                                                String.join(", ", toAdd)));
                        }
                }
                return sb.toString();
        }

        // ── Inner record for occasion rules ───────────────────────────────────────

        /**
         * Encapsulates a set of style rules for a specific occasion.
         *
         * @param itemsToAvoid   clothing categories that don't fit this occasion
         * @param preferredItems clothing categories that enhance this occasion look
         * @param minFormality   minimum formality score required for this occasion
         */
        private record OccasionRule(
                        List<String> itemsToAvoid,
                        List<String> preferredItems,
                        double minFormality) {
        }
}
