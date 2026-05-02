package com.fashionai.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;

import java.time.Instant;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Document(collection = "recommendation_feedback")
public class RecommendationFeedback {

    @Id
    private String id;

    private String userId;
    private String outfitId;
    private String style;
    private String action; // "like", "dislike", "click"
    private String occasion;

    @CreatedDate
    private Instant createdAt;
}
