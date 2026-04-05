package com.fashionai.controller;

import com.fashionai.model.ClothingItem;
import com.fashionai.model.UpgradeResult;
import com.fashionai.service.OutfitUpgradeService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

/**
 * REST controller for outfit upgrade suggestions.
 *
 * Endpoints:
 *   POST /api/upgrade-outfit  — Analyze and upgrade an outfit given detected items + occasion
 *   GET  /api/upgrade-history — Retrieve recent upgrade history
 */
@Slf4j
@RestController
@RequestMapping("/api")
@RequiredArgsConstructor
@CrossOrigin(origins = "*")
public class UpgradeController {

    private final OutfitUpgradeService upgradeService;

    /**
     * Accepts already-detected clothing items and an occasion, returns upgrade suggestions.
     *
     * This endpoint is useful when the client has already called /detect-clothes and wants
     * to get upgrade suggestions without re-uploading the image.
     *
     * Expected request body:
     * <pre>
     * {
     *   "detectedItems": [
     *     { "category": "t-shirt", "style": "casual", "formalityScore": 0.2 },
     *     { "category": "jeans", "style": "casual", "formalityScore": 0.25 }
     *   ],
     *   "occasion": "party"
     * }
     * </pre>
     *
     * @param body JSON body with detectedItems array and occasion string
     * @return UpgradeResult with suggestions and complementary recommendations
     */
    @PostMapping("/upgrade-outfit")
    public ResponseEntity<UpgradeResult> upgradeOutfit(@RequestBody Map<String, Object> body) {
        String occasion = (String) body.getOrDefault("occasion", "casual");

        @SuppressWarnings("unchecked")
        List<Map<String, Object>> rawItems = (List<Map<String, Object>>) body.get("detectedItems");

        log.info("POST /api/upgrade-outfit — occasion='{}', items={}", occasion,
                rawItems != null ? rawItems.size() : 0);

        if (rawItems == null || rawItems.isEmpty()) {
            return ResponseEntity.badRequest().build();
        }

        // Map raw JSON items to ClothingItem objects
        List<ClothingItem> items = rawItems.stream()
                .map(this::mapToClothingItem)
                .toList();

        // Run upgrade pipeline (without raw image bytes — features will use mock)
        UpgradeResult result = upgradeService.upgradeOutfit(items, occasion, new byte[0]);
        return ResponseEntity.ok(result);
    }

    /**
     * Returns the 20 most recent upgrade analyses.
     * Allows the frontend to display a history feed.
     */
    @GetMapping("/upgrade-history")
    public ResponseEntity<Map<String, Object>> getHistory() {
        log.info("GET /api/upgrade-history");
        List<UpgradeResult> history = upgradeService.getRecentHistory();
        return ResponseEntity.ok(Map.of(
                "history", history,
                "count", history.size()
        ));
    }

    /**
     * Maps a raw JSON map (from request body) into a ClothingItem.
     */
    private ClothingItem mapToClothingItem(Map<String, Object> raw) {
        return ClothingItem.builder()
                .category((String) raw.getOrDefault("category", "unknown"))
                .style((String) raw.getOrDefault("style", "casual"))
                .formalityScore(
                        raw.containsKey("formalityScore")
                                ? ((Number) raw.get("formalityScore")).doubleValue()
                                : 0.3)
                .confidence(
                        raw.containsKey("confidence")
                                ? ((Number) raw.get("confidence")).doubleValue()
                                : 1.0)
                .build();
    }
}
