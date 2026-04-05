package com.fashionai.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;

import java.util.List;

/**
 * Represents a single clothing item detected in an outfit image.
 * Stored in MongoDB and used throughout the detection → recommendation pipeline.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Document(collection = "clothing_items")
public class ClothingItem {

    @Id
    private String id;

    /** Category label from YOLO detection (e.g., "shirt", "jeans", "boots") */
    private String category;

    /** Detection confidence score from YOLO (0.0–1.0) */
    private double confidence;

    /** Bounding box: [x1, y1, x2, y2] in pixel coordinates */
    private List<Integer> boundingBox;

    /** 2048-dimensional ResNet50 feature vector for similarity computation */
    private List<Double> featureVector;

    /** Dominant color extracted from the item region (e.g., "navy blue") */
    private String dominantColor;

    /** Style classification: casual | smart-casual | formal | sporty */
    private String style;

    /** Formality score 0.0 (very casual) → 1.0 (very formal) */
    private double formalityScore;

    /** Season suitability: spring | summer | autumn | winter | all */
    private String season;

    /** Source image S3/local path for reference */
    private String imageSource;
}
