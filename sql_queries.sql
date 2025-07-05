-- 
CREATE TABLE machine (
    machine_id UNIQUEIDENTIFIER PRIMARY KEY,
    created_at DATETIME NOT NULL
);

-- created table for chat
CREATE TABLE chat_table (
    chat_id NVARCHAR(255) PRIMARY KEY,
    content_type NVARCHAR(MAX),
    content NVARCHAR(MAX),
    created_at DATETIME,
    content_id NVARCHAR(255), 
    machine_id NVARCHAR(255),
    chat_group_id NVARCHAR(255), 
    sources NVARCHAR(MAX) 
);

-- 
CREATE TABLE coupon_details (
    brand_name NVARCHAR(255),
    activation_start_date DATETIME,
    activation_end_date DATETIME,
    coupon_code NVARCHAR(255), 
    coupon_group_id NVARCHAR(255)
);


-- 
INSERT INTO coupon_details (brand_name, activation_start_date, activation_end_date, coupon_code, coupon_group_id)
VALUES
('Nike', '2025-07-01 00:00:00', '2025-07-31 23:59:59', 'NIKE20OFF', 'SUMMER2025'),
('Adidas', '2025-07-05 09:00:00', '2025-08-15 18:00:00', 'ADIDASFREESHIP', 'SUMMER2025'),
('Starbucks', '2025-06-15 08:00:00', '2025-07-15 22:00:00', 'COFFEEBREAK', 'JULYPROMOS'),
('Amazon', '2025-07-10 00:00:00', '2025-07-12 23:59:59', 'PRIME24SALE', 'PRIME_DAY'),
('Sephora', '2025-08-01 00:00:00', '2025-08-31 23:59:59', 'BEAUTYBLAST', 'AUGUSTDEALS'),
('Zara', '2025-07-01 10:00:00', '2025-07-20 20:00:00', 'ZARAFF25', 'FASHIONFRENZY'),
('Google Store', '2025-07-01 00:00:00', '2025-09-30 23:59:59', 'PIXELBUDSDEAL', 'TECHSAVINGS'),
('McDonalds', '2025-07-01 06:00:00', '2025-07-07 23:59:59', 'BIGMACMONDAY', 'WEEKLYSPECIALS'),
('Best Buy', '2025-07-08 09:00:00', '2025-07-14 21:00:00', 'TECHTUESDAY', 'WEEKLYSPECIALS'),
('Target', '2025-07-01 00:00:00', '2025-07-31 23:59:59', 'TARGETJULY', 'JULYPROMOS');

-- 
CREATE TABLE coupon (
    coupon_group_id NVARCHAR(255), 
    coupon_keywords NVARCHAR(MAX), 
    product_url NVARCHAR(MAX) 
);

-- 
INSERT INTO coupon (coupon_group_id, coupon_keywords, product_url)
VALUES
('SUMMER2025', 'summer, discount, shoes, apparel', 'https://www.nike.com/summer-sale'),
('JULYPROMOS', 'coffee, drinks, special, july', 'https://www.starbucks.com/july-offers'),
('PRIME_DAY', 'amazon, prime, electronics, deals', 'https://www.amazon.com/prime-day-deals'),
('AUGUSTDEALS', 'beauty, makeup, skincare, august', 'https://www.sephora.com/august-specials'),
('FASHIONFRENZY', 'fashion, clothes, dresses, accessories', 'https://www.zara.com/new-arrivals'),
('TECHSAVINGS', 'tech, electronics, gadgets, pixel', 'https://store.google.com/deals'),
('WEEKLYSPECIALS', 'food, fast food, restaurant, weekly', 'https://www.mcdonalds.com/weekly-deals'),
('HOLIDAY2025', 'holiday, gifts, season', 'https://www.macys.com/holiday-gifts');
