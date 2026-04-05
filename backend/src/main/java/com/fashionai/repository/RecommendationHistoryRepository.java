package com.fashionai.repository;

import com.fashionai.model.UpgradeResult;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

/**
 * Spring Data MongoDB repository for storing outfit upgrade/recommendation history.
 * Allows users to revisit past analyses.
 */
@Repository
public interface RecommendationHistoryRepository extends MongoRepository<UpgradeResult, String> {

    /** Find the most recent N upgrade results (history feed) */
    List<UpgradeResult> findTop20ByOrderByCreatedAtDesc();

    /** Find all upgrade results for a specific occasion */
    List<UpgradeResult> findByOccasion(String occasion);
}
