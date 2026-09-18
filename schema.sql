-- Database Schema for Mobile Price Prediction System
CREATE DATABASE IF NOT EXISTS mobile_price_db;

USE mobile_price_db;

CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    ram_gb INT NOT NULL,
    storage_gb INT NOT NULL,
    battery_mah INT NOT NULL,
    primary_camera_mp INT NOT NULL,
    front_camera_mp INT NOT NULL,
    cpu_speed_ghz FLOAT NOT NULL,
    cpu_cores INT NOT NULL,
    screen_size_inch FLOAT NOT NULL,
    has_5g BOOLEAN NOT NULL DEFAULT 0,
    predicted_price DECIMAL(10, 2) NOT NULL,
    price_tier VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
