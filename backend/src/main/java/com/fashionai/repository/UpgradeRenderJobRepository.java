package com.fashionai.repository;

import com.fashionai.model.UpgradeRenderJob;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface UpgradeRenderJobRepository extends MongoRepository<UpgradeRenderJob, String> {
}
