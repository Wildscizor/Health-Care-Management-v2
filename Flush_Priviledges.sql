CREATE DATABASE healthcare CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'ACE'@'localhost' IDENTIFIED BY 'siddu';
GRANT ALL PRIVILEGES ON healthcare.* TO 'ACE'@'localhost';
FLUSH PRIVILEGES;
