package com.fashionai.repository;

import com.fashionai.model.UserPreference;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

/**
 * Spring Data MongoDB repository for user preference documents.
 */
@Repository
public interface UserPreferenceRepository extends MongoRepository<UserPreference, String> {

    /** Find preferences by the application-level user ID (not the MongoDB _id) */
    Optional<UserPreference> findByUserId(String userId);

    /** Check if a user profile exists */
    boolean existsByUserId(String userId);
}
