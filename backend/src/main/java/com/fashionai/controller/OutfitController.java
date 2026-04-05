package com.fashionai.controller;

import com.fashionai.model.ClothingItem;
import com.fashionai.model.UpgradeResult;
import com.fashionai.service.OutfitDetectionService;
import com.fashionai.service.OutfitUpgradeService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * REST controller that handles outfit image uploads.
 *
 * Endpoints:
 * POST /api/upload-outfit — Full pipeline: detect → extract → upgrade →
 * recommend
 * POST /api/detect-clothes — Clothing detection only (without upgrade)
 */
@Slf4j
@RestController
@RequestMapping("/api")
@RequiredArgsConstructor
@CrossOrigin(origins = "*")
public class OutfitController {

    private final OutfitDetectionService detectionService;
    private final OutfitUpgradeService upgradeService;

    /**
     * Full pipeline endpoint.
     *
     * Accepts an outfit image and occasion, runs the complete workflow:
     * detection → feature extraction → compatibility → upgrade → recommendations
     *
     * @param image    multipart image file (supported: jpg, png, webp)
     * @param occasion target occasion string (e.g., "party", "work", "casual")
     * @return complete UpgradeResult JSON
     */
    @PostMapping(value = "/upload-outfit", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<?> uploadOutfit(
            @RequestParam("image") MultipartFile image,
            @RequestParam(value = "occasion", defaultValue = "casual") String occasion) {

        log.info("POST /api/upload-outfit — file='{}', size={}KB, occasion='{}'",
                image.getOriginalFilename(),
                image.getSize() / 1024,
                occasion);

        try {
            // ── Step 1: Detect clothing items ─────────────────────────────────
            List<ClothingItem> detectedItems = detectionService.detectClothing(image);

            if (detectedItems.isEmpty()) {
                return ResponseEntity.badRequest().body(Map.of(
                        "message",
                        "No clothing items detected. The YOLO model could not identify clear clothing regions in the image. Please try an image with more visible or separated clothing items."));
            }

            // ── Step 2: Run full upgrade pipeline ────────────────────────────
            UpgradeResult result = upgradeService.upgradeOutfit(
                    detectedItems, occasion, image.getBytes());

            return ResponseEntity.ok(result);

        } catch (IOException e) {
            log.error("Error processing outfit image: {}", e.getMessage(), e);
            return ResponseEntity.internalServerError().build();
        }
    }

    /**
     * Clothing detection only — returns detected items without upgrade logic.
     * Useful for quickly testing detection without the full pipeline.
     *
     * @param image multipart image file
     * @return list of detected ClothingItem objects
     */
    @PostMapping(value = "/detect-clothes", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<Map<String, Object>> detectClothes(
            @RequestParam("image") MultipartFile image) {

        log.info("POST /api/detect-clothes — file='{}'", image.getOriginalFilename());

        try {
            List<ClothingItem> items = detectionService.detectClothing(image);
            return ResponseEntity.ok(Map.of(
                    "detected_items", items,
                    "item_count", items.size(),
                    "status", "success"));

        } catch (IOException e) {
            log.error("Detection error: {}", e.getMessage());
            return ResponseEntity.internalServerError()
                    .body(Map.of("error", e.getMessage(), "status", "error"));
        }
    }

    /**
     * Health check endpoint for this controller.
     */
    @GetMapping("/health")
    public ResponseEntity<Map<String, String>> health() {
        return ResponseEntity.ok(Map.of("status", "UP", "service", "AI Fashion Recommender"));
    }
}
