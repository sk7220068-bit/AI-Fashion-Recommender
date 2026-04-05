package com.fashionai.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.lang.NonNull;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

/**
 * Global CORS and web MVC configuration.
 * Allows the React frontend (localhost:5173) to call the Spring Boot API (localhost:8080).
 */
@Configuration
public class WebConfig implements WebMvcConfigurer {

    @Override
    public void addCorsMappings(@NonNull CorsRegistry registry) {
        registry.addMapping("/api/**")
                // Allow React dev server and production builds
                .allowedOrigins(
                        "http://localhost:5173",  // Vite dev server
                        "http://localhost:3000",  // CRA dev server
                        "http://localhost:8080"   // Same-origin (Thymeleaf)
                )
                .allowedMethods("GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH")
                .allowedHeaders("*")
                .allowCredentials(false)
                .maxAge(3600);
    }
}
