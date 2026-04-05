package com.fashionai.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;
import org.springframework.data.mongodb.core.index.Indexed;

import java.util.List;

/**
 * Stores a user's persistent fashion preferences used to personalize recommendations.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Document(collection = "user_preferences")
public class UserPreference {

    @Id
    private String id;

    /** Unique user identifier */
    @Indexed(unique = true)
    private String userId;

    /** User's display name */
    private String name;

    /** Preferred style categories (e.g., ["casual", "streetwear"]) */
    @Builder.Default
    private List<String> preferredStyles = List.of();

    /** Frequently attended occasions (e.g., ["work", "casual", "formal"]) */
    @Builder.Default
    private List<String> frequentOccasions = List.of();

    /** Preferred color palette (e.g., ["navy", "white", "earth-tones"]) */
    @Builder.Default
    private List<String> preferredColors = List.of();

    /** Avoided items or styles */
    @Builder.Default
    private List<String> avoidedItems = List.of();

    /** Body type (used to filter fits): slim | regular | plus | athletic */
    private String bodyType;

    /** Season preferences */
    @Builder.Default
    private List<String> preferredSeasons = List.of();

    /** IDs of outfits the user has liked */
    @Builder.Default
    private List<String> likedOutfitIds = List.of();

    /** IDs of outfits the user has disliked */
    @Builder.Default
    private List<String> dislikedOutfitIds = List.of();
}
