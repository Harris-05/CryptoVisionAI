create database crypto
use crypto

CREATE TABLE CryptoCoins (
    id VARCHAR(100) PRIMARY KEY,       
    name VARCHAR(200),                      
    symbol VARCHAR(100),                     
    currentprice FLOAT NOT NULL,            
    marketcap BIGINT NOT NULL,              
    marketrank INT NOT NULL,                
    highesttday FLOAT,                      
    lowesttday FLOAT,                       
    pricechange FLOAT,                      
    total_volume BIGINT,                    
    circulating_supply FLOAT,               
    lastupdated DATETIME                    
);

CREATE TABLE pricehistory (
    id INT PRIMARY KEY IDENTITY(1,1),
    cid VARCHAR(100),
    price FLOAT,
    marketrank INT,
    marketcap FLOAT,
    circulating_supply FLOAT,  
    timeentery DATETIME,
    FOREIGN KEY (cid) REFERENCES CryptoCoins(id)
);
select * from cryptocoins order by marketrank asc
select * from cryptocoins order by currentprice desc
select * from pricehistory
SELECT timeentery, price, marketrank
    FROM pricehistory
    WHERE cid = 'bitcoin'
    ORDER BY timeentery ASC

select * from pricehistory where cid = 'bitcoin'