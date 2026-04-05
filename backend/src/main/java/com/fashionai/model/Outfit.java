package com.fashionai.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.data.annotation.Id;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.mongodb.core.mapping.Document;
import org.springframework.data.mongodb.core.index.Indexed;

import java.time.Instant;
import java.util.List;

/**
 * Represents a complete outfit — a collection of clothing items
 * associated with an occasion, style, and metadata for recommendations.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Document(collection = "outfits")
public class Outfit {

    @Id
    private String id;

    /** Human-readable outfit name (e.g., "Summer Casual Look") */
    private String name;

    /** Description of the outfit */
    private String description;

    /** List of clothing items that make up this outfit */
    private List<ClothingItem> items;

    /** Occasion tags (e.g., party, work, casual, formal, date) */
    @Indexed
    private List<String> occasions;

    /** Style category: casual | smart-casual | formal | sporty | streetwear */
    @Indexed
    private String style;

    /** Color palette of the overall outfit (e.g., ["navy", "white", "grey"]) */
    private List<String> colorPalette;

    /** Season suitability: spring | summer | autumn | winter | all */
    private String season;

    /** Overall formality score 0.0 → 1.0 */
    private double formalityScore;

    /**
     * Aggregate feature vector — average of all item feature vectors.
     * Used for cosine-similarity-based recommendations.
     */
    private List<Double> aggregateFeatureVector;

    /** Tags for search and filtering */
    private List<String> tags;

    /** Popularity/rating score */
    private double popularityScore;

    /** Timestamp when this outfit was added to the dataset */
    @CreatedDate
    private Instant createdAt;

    /** Source identifier ("dataset" | "user-upload" | "ai-generated") */
    private String source;
}
