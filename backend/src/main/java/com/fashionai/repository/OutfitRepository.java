package com.fashionai.repository;

import com.fashionai.model.Outfit;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.data.mongodb.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.List;

/**
 * Spring Data MongoDB repository for Outfit documents.
 * Provides CRUD and custom query methods for the outfits collection.
 */
@Repository
public interface OutfitRepository extends MongoRepository<Outfit, String> {

    /** Find all outfits matching a specific style category */
    List<Outfit> findByStyle(String style);

    /** Find outfits tagged with a given occasion */
    List<Outfit> findByOccasionsContaining(String occasion);

    /** Find outfits matching both style and occasion */
    List<Outfit> findByStyleAndOccasionsContaining(String style, String occasion);

    /** Find outfits with a formality score above the given threshold */
    List<Outfit> findByFormalityScoreGreaterThanEqual(double formalityScore);

    /** Find outfits by season suitability */
    List<Outfit> findBySeason(String season);

    /** Find dataset-sourced outfits only */
    @Query("{ 'source': 'dataset' }")
    List<Outfit> findDatasetOutfits();

    /** Find the top N most popular outfits */
    List<Outfit> findTop10ByOrderByPopularityScoreDesc();
}
