package com.fashionai.controller;

import com.fashionai.model.OutfitRecommendation;
import com.fashionai.service.RecommendationService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

/**
 * REST controller for outfit recommendation queries.
 *
 * Endpoints:
 *   POST /api/recommend-outfit   — Recommend outfits by occasion + style (no image required)
 *   GET  /api/recommend-outfit   — Same as POST but via query params
 */
@Slf4j
@RestController
@RequestMapping("/api")
@RequiredArgsConstructor
@CrossOrigin(origins = "*")
public class RecommendationController {

    private final RecommendationService recommendationService;

    /**
     * Get outfit recommendations by providing occasion and style preferences.
     * Does not require an image — uses metadata-based filtering + popularity scoring.
     *
     * @param body JSON body with keys: occasion (string), style (string)
     * @return list of OutfitRecommendation objects
     */
    @PostMapping("/recommend-outfit")
    public ResponseEntity<Map<String, Object>> recommendOutfit(
            @RequestBody Map<String, String> body) {

        String occasion = body.getOrDefault("occasion", "casual");
        String style    = body.getOrDefault("style", null);

        log.info("POST /api/recommend-outfit — occasion='{}', style='{}'", occasion, style);

        List<OutfitRecommendation> recommendations =
                recommendationService.recommendByOccasionAndStyle(occasion, style);

        return ResponseEntity.ok(Map.of(
                "recommendations", recommendations,
                "count", recommendations.size(),
                "occasion", occasion,
                "style", style != null ? style : "any"
        ));
    }

    /**
     * GET variant of the recommendation endpoint for easy browser testing.
     */
    @GetMapping("/recommend-outfit")
    public ResponseEntity<Map<String, Object>> recommendOutfitGet(
            @RequestParam(value = "occasion", defaultValue = "casual") String occasion,
            @RequestParam(value = "style", required = false) String style) {

        log.info("GET /api/recommend-outfit — occasion='{}', style='{}'", occasion, style);

        List<OutfitRecommendation> recommendations =
                recommendationService.recommendByOccasionAndStyle(occasion, style);

        return ResponseEntity.ok(Map.of(
                "recommendations", recommendations,
                "count", recommendations.size()
        ));
    }
}
