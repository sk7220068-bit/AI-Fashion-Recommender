package com.fashionai;

import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;

/**
 * AI Fashion Recommender — Spring Boot Application Entry Point.
 *
 * Full workflow:
 * User uploads outfit image
 * → /api/upload-outfit (OutfitController)
 * → OutfitDetectionService (YOLO via Python ML service)
 * → FeatureExtractionService (ResNet50 via Python ML service)
 * → CompatibilityService (pairwise scoring)
 * → OutfitUpgradeEngine (rule-based occasion upgrade)
 * → RecommendationService (cosine similarity rankings)
 * → UpgradeResult JSON returned to React frontend
 *
 * Run: mvn spring-boot:run
 * API: http://localhost:8080/api
 * UI: http://localhost:5173 (React dev server)
 */
@Slf4j
@SpringBootApplication
public class FashionAIApplication {

    public static void main(String[] args) {
        SpringApplication.run(FashionAIApplication.class, args);
    }

    @EventListener(ApplicationReadyEvent.class)
    public void onReady() {
        log.info("╔══════════════════════════════════════════════════╗");
        log.info("║   AI Fashion Recommender is UP and running!      ║");
        log.info("║   API:  http://localhost:8080/api                ║");
        log.info("║   Docs: http://localhost:8080/api/health         ║");
        log.info("╚══════════════════════════════════════════════════╝");
    }

}
