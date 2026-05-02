package com.fashionai.repository;

import com.fashionai.model.RecommendationFeedback;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface RecommendationFeedbackRepository extends MongoRepository<RecommendationFeedback, String> {

    long countByUserIdAndOutfitIdAndAction(String userId, String outfitId, String action);

    long countByUserIdAndStyleAndAction(String userId, String style, String action);
}
