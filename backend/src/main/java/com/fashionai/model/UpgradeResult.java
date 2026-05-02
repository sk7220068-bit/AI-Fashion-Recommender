package com.fashionai.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.data.annotation.Id;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.mongodb.core.mapping.Document;

import java.time.Instant;
import java.util.List;

/**
 * The final result returned to the user after the outfit upgrade pipeline.
 * Contains the original detected items, upgrade suggestions, and compatibility analysis.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Document(collection = "upgrade_results")
public class UpgradeResult {

    @Id
    private String id;

    /** The occasion provided by the user (e.g., "party") */
    private String occasion;

    /** Clothing items detected in the uploaded image */
    private List<ClothingItem> detectedItems;

    /** Overall compatibility score of the uploaded outfit (0.0 → 1.0) */
    private double overallCompatibilityScore;

    /** Items that should be replaced with better alternatives */
    private List<String> itemsToReplace;

    /** Specific upgrade suggestions (e.g., "Replace sneakers with leather boots") */
    private List<String> upgradeSuggestions;

    /** Items to add to the outfit (e.g., "Add a navy blazer") */
    private List<String> itemsToAdd;

    /** Human-readable summary of the upgrade analysis */
    private String upgradeSummary;

    /** AI-generated outfit description (if OpenAI integration enabled) */
    private String aiGeneratedDescription;

    /** Compatible outfit recommendations alongside the upgrade */
    private List<OutfitRecommendation> recommendations;

    /** Rendered visual preview URL (or data URI) for the upgraded outfit */
    private String upgradedImageUrl;

    /** Alternative rendered upgrade previews */
    private List<String> upgradedImageAlternatives;

    /** Rendering status: ready | pending | failed */
    private String renderStatus;

    /** When this result was generated */
    @CreatedDate
    private Instant createdAt;
}
