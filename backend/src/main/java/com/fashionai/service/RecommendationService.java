package com.fashionai.service;

import com.fashionai.model.ClothingItem;
import com.fashionai.model.Outfit;
import com.fashionai.model.OutfitRecommendation;
import com.fashionai.recommendation.CosineSimilarityEngine;
import com.fashionai.repository.OutfitRepository;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Service;

import java.io.InputStreamReader;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.Collections;
import java.util.stream.Collectors;

/**
 * Content-based outfit recommendation service.
 *
 * Pipeline:
 *  1. Load outfits from MongoDB (seeded from CSV on first run)
 *  2. Compute cosine similarity between query vector and each dataset outfit
 *  3. Apply occasion-boost weighting
 *  4. Sort by composite rank score and return top-N results
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class RecommendationService {

    private final OutfitRepository outfitRepository;
    private final CosineSimilarityEngine cosineSimilarity;

    @Value("${dataset.outfits-csv-path}")
    private Resource outfitsCsvResource;

    @Value("${recommendation.similarity-threshold:0.5}")
    private double similarityThreshold;

    @Value("${recommendation.max-results:10}")
    private int maxResults;

    /**
     * Seeds the database from the CSV dataset on application startup (if empty).
     * Ensures outfits are available for recommendation without manual setup.
     */
    @PostConstruct
    public void seedDatasetIfEmpty() {
        long count = outfitRepository.count();
        if (count == 0) {
            log.info("Outfit collection empty — seeding from CSV dataset");
            try {
                List<Outfit> outfits = Collections.unmodifiableList(loadOutfitsFromCsv());
                outfitRepository.saveAll(outfits);
                log.info("Seeded {} outfits from dataset", outfits.size());
            } catch (Exception e) {
                log.error("Failed to seed dataset: {}", e.getMessage());
            }
        } else {
            log.info("Outfit collection has {} records — skipping seed", count);
        }
    }

    /**
     * Returns outfit recommendations based on a query feature vector and occasion.
     *
     * @param queryVector  aggregate feature vector of the uploaded outfit
     * @param occasion     target occasion filter (can be null for broad search)
     * @param style        target style filter (can be null)
     * @return             list of OutfitRecommendation sorted by ranking score
     */
    public List<OutfitRecommendation> recommend(List<Double> queryVector,
                                                  String occasion,
                                                  String style) {
        log.info("Generating recommendations for occasion='{}', style='{}'", occasion, style);

        // Load all dataset outfits for similarity computation
        List<Outfit> candidates = outfitRepository.findDatasetOutfits();

        if (candidates.isEmpty()) {
            log.warn("No dataset outfits found — returning empty recommendations");
            return List.of();
        }

        List<OutfitRecommendation> recommendations = new ArrayList<>();

        for (Outfit outfit : candidates) {
            // Skip outfits without feature vectors
            if (outfit.getAggregateFeatureVector() == null ||
                    outfit.getAggregateFeatureVector().isEmpty()) continue;

            double similarity = cosineSimilarity.computeSimilarity(
                    queryVector, outfit.getAggregateFeatureVector());

            // Filter out outfits below the similarity threshold
            if (similarity < similarityThreshold) continue;

            // Occasion compatibility boost (rewards exact occasion match)
            double occasionBoost = computeOccasionBoost(outfit, occasion);

            // Style compatibility boost
            double styleBoost = computeStyleBoost(outfit, style);

            // Composite ranking score
            double rankingScore = (similarity * 0.6) + (occasionBoost * 0.3) + (styleBoost * 0.1);

            recommendations.add(OutfitRecommendation.builder()
                    .outfit(outfit)
                    .similarityScore(similarity)
                    .occasionCompatibilityScore(occasionBoost)
                    .rankingScore(rankingScore)
                    .recommendationReason(buildReason(outfit, occasion, similarity))
                    .build());
        }

        // Sort by ranking score descending, take top-N
        recommendations.sort(Comparator.comparingDouble(OutfitRecommendation::getRankingScore).reversed());
        List<OutfitRecommendation> topResults = recommendations.stream()
                .limit(maxResults)
                .collect(Collectors.toList());

        // Assign rank positions
        for (int i = 0; i < topResults.size(); i++) {
            topResults.get(i).setRank(i + 1);
        }

        log.info("Returning {} recommendations (from {} candidates)", topResults.size(), candidates.size());
        return topResults;
    }

    /**
     * Simplified recommendation for cases where no feature vector is available.
     * Falls back to occasion and style filtering only.
     */
    public List<OutfitRecommendation> recommendByOccasionAndStyle(String occasion, String style) {
        List<Outfit> candidates;
        if (occasion != null && style != null) {
            candidates = outfitRepository.findByStyleAndOccasionsContaining(style, occasion);
        } else if (occasion != null) {
            candidates = outfitRepository.findByOccasionsContaining(occasion);
        } else if (style != null) {
            candidates = outfitRepository.findByStyle(style);
        } else {
            candidates = outfitRepository.findTop10ByOrderByPopularityScoreDesc();
        }

        return candidates.stream()
                .sorted(Comparator.comparingDouble(Outfit::getPopularityScore).reversed())
                .limit(maxResults)
                .map(outfit -> OutfitRecommendation.builder()
                        .outfit(outfit)
                        .similarityScore(1.0)
                        .occasionCompatibilityScore(1.0)
                        .rankingScore(outfit.getPopularityScore())
                        .recommendationReason(String.format("Popular %s outfit for %s", outfit.getStyle(), occasion))
                        .build())
                .collect(Collectors.toList());
    }

    // ── Private helpers ────────────────────────────────────────────────────────

    private double computeOccasionBoost(Outfit outfit, String occasion) {
        if (occasion == null || outfit.getOccasions() == null) return 0.5;
        return outfit.getOccasions().contains(occasion.toLowerCase()) ? 1.0 : 0.3;
    }

    private double computeStyleBoost(Outfit outfit, String style) {
        if (style == null || outfit.getStyle() == null) return 0.5;
        return outfit.getStyle().equalsIgnoreCase(style) ? 1.0 : 0.4;
    }

    private String buildReason(Outfit outfit, String occasion, double similarity) {
        return String.format("%.0f%% visual match — %s style, suitable for %s",
                similarity * 100,
                outfit.getStyle(),
                occasion != null ? occasion : "multiple occasions");
    }

    /**
     * Parses the outfits.csv file from classpath and builds Outfit domain objects.
     * Expected CSV columns: id,name,style,occasions,items,color_palette,season,formality_score,feature_vector
     */
    private List<Outfit> loadOutfitsFromCsv() throws Exception {
        List<Outfit> outfits = new ArrayList<>();

        try (Reader reader = new InputStreamReader(outfitsCsvResource.getInputStream(), StandardCharsets.UTF_8);
             CSVParser parser = CSVFormat.DEFAULT.builder()
                     .setHeader()
                     .setSkipHeaderRecord(true)
                     .setTrim(true)
                     .build()
                     .parse(reader)) {

            for (CSVRecord record : parser) {
                try {
                    Outfit outfit = parseOutfitRecord(record);
                    outfits.add(outfit);
                } catch (Exception e) {
                    log.warn("Skipping malformed CSV record {}: {}", record.getRecordNumber(), e.getMessage());
                }
            }
        }
        return outfits;
    }

    /**
     * Parses a single CSV record into an Outfit object.
     * The feature_vector column contains a pipe-separated list of doubles.
     */
    private Outfit parseOutfitRecord(CSVRecord record) {
        String featureVectorStr = record.get("feature_vector");
        List<Double> featureVector = parseFeatureVector(featureVectorStr);

        String occasions = record.get("occasions");
        List<String> occasionList = Arrays.stream(occasions.split("\\|"))
                .map(String::trim)
                .collect(Collectors.toList());

        String colorPalette = record.get("color_palette");
        List<String> colorList = Arrays.stream(colorPalette.split("\\|"))
                .map(String::trim)
                .collect(Collectors.toList());

        String itemsStr = record.get("items");
        List<ClothingItem> items = Arrays.stream(itemsStr.split("\\|"))
                .map(String::trim)
                .map(cat -> ClothingItem.builder().category(cat).confidence(1.0).build())
                .collect(Collectors.toList());

        return Outfit.builder()
                .name(record.get("name"))
                .style(record.get("style"))
                .occasions(occasionList)
                .items(items)
                .colorPalette(colorList)
                .season(record.get("season"))
                .formalityScore(Double.parseDouble(record.get("formality_score")))
                .aggregateFeatureVector(featureVector)
                .popularityScore(4.0 + Math.random())
                .source("dataset")
                .build();
    }

    private List<Double> parseFeatureVector(String vectorStr) {
        if (vectorStr == null || vectorStr.isBlank()) return generateDefaultVector();
        try {
            return Arrays.stream(vectorStr.split("\\|"))
                    .map(Double::parseDouble)
                    .collect(Collectors.toList());
        } catch (NumberFormatException e) {
            return generateDefaultVector();
        }
    }

    /** Generates a random 64-dim placeholder feature vector for CSV entries without one */
    private List<Double> generateDefaultVector() {
        Random r = new Random();
        List<Double> v = new ArrayList<>(64);
        for (int i = 0; i < 64; i++) v.add(r.nextDouble());
        return v;
    }
}
