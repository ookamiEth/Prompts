# The Ultimate DuneSQL Performance Guide for DeFi Protocol Analysis

## Table of Contents
1. [Introduction](#introduction)
2. [Understanding Dune's Data Architecture](#understanding-dunes-data-architecture)
3. [Core DeFi Tables and Data Sources](#core-defi-tables-and-data-sources)
4. [Smart Contract Data Analysis Fundamentals](#smart-contract-data-analysis-fundamentals)
5. [Query Performance Optimization Principles](#query-performance-optimization-principles)
6. [Advanced DeFi Analysis Patterns](#advanced-defi-analysis-patterns)
7. [Token Movement and Transfer Analysis](#token-movement-and-transfer-analysis)
8. [Common Performance Pitfalls and Solutions](#common-performance-pitfalls-and-solutions)
9. [Practical Query Examples](#practical-query-examples)
10. [Advanced Optimization Techniques](#advanced-optimization-techniques)
11. [Troubleshooting and Debugging](#troubleshooting-and-debugging)

---

## Introduction

Welcome to the most comprehensive guide for writing high-performance DuneSQL queries for DeFi protocol analysis. This guide is designed specifically for entry-level junior analysts who need extremely detailed instructions to properly analyze smart contracts, token movements, and DeFi protocols.

**Key Learning Objectives:**
- Master Dune's three-tier data architecture for optimal query performance
- Understand how to efficiently analyze smart contract interactions
- Write blazingly fast queries for token transfer analysis
- Avoid common performance pitfalls that slow down your queries
- Implement advanced patterns for complex DeFi protocol analysis

**Performance First Mindset:**
Every query you write should be optimized for speed and efficiency. In this guide, you'll learn that choosing the right table, applying filters correctly, and understanding data structures can make the difference between a 2-second query and a 2-minute timeout.

---

## Understanding Dune's Data Architecture

### The Three-Tier Architecture: Raw → Decoded → Curated

Dune organizes blockchain data into three distinct layers, and understanding this is **CRITICAL** for writing performant queries:

#### 1. Raw Data Layer (Lowest Performance)
```sql
-- RAW TABLES (USE SPARINGLY)
ethereum.blocks           -- Block metadata
ethereum.transactions     -- Transaction details  
ethereum.logs            -- Event logs (encoded)
ethereum.traces          -- Internal transaction traces
```

**When to use:** Only when you need data that hasn't been decoded yet, or when you're doing very specific low-level blockchain analysis.

**Performance Impact:** ⚠️ **SLOW** - These tables are massive and require manual decoding.

#### 2. Decoded Data Layer (High Performance)
```sql
-- DECODED TABLES (PREFER THESE)
uniswap_v3_ethereum.pool_evt_swap     -- Specific contract events
aave_v2_ethereum.lendingpool_call_*   -- Contract function calls
erc20_ethereum.evt_transfer           -- All ERC20 transfers
```

**When to use:** 95% of your DeFi analysis should use these tables.

**Performance Impact:** ✅ **FAST** - Pre-processed, indexed, and human-readable.

#### 3. Curated Data Layer (Highest Performance)
```sql
-- CURATED TABLES (FASTEST)
dex.trades              -- All DEX trades across protocols
nft.trades             -- All NFT marketplace trades
tokens.erc20           -- Token metadata and prices
prices.usd             -- Price feeds
```

**When to use:** When you need cross-protocol analysis or standardized metrics.

**Performance Impact:** 🚀 **FASTEST** - Optimized, aggregated, and ready for analysis.

### **GOLDEN RULE #1: Always Use the Highest Layer Possible**

```sql
-- ❌ WRONG - Using raw logs (slow)
SELECT * FROM ethereum.logs 
WHERE contract_address = 0xa0b86a33e6a8e0b57c5e678b15dce2ffec3e9de5

-- ✅ CORRECT - Using decoded table (fast)
SELECT * FROM uniswap_v2_ethereum.pair_evt_swap
WHERE contract_address = 0xa0b86a33e6a8e0b57c5e678b15dce2ffec3e9de5

-- 🚀 BEST - Using curated table (fastest)
SELECT * FROM dex.trades 
WHERE project = 'uniswap' 
  AND version = '2'
  AND pool_address = 0xa0b86a33e6a8e0b57c5e678b15dce2ffec3e9de5
```

---

## Core DeFi Tables and Data Sources

### Essential Curated Tables for DeFi Analysis

#### 1. `dex.trades` - The Most Important Table for DeFi

This table contains **ALL** decentralized exchange trades across **40+ blockchains** and **100+ protocols**.

**Table Structure:**
```sql
SELECT 
    blockchain,           -- Chain where trade occurred
    project,             -- DEX protocol (uniswap, sushiswap, etc.)
    version,             -- Protocol version (v2, v3, etc.)
    block_date,          -- Date of trade (indexed - FAST filtering)
    block_time,          -- Timestamp of trade (indexed - FAST filtering)
    token_bought_symbol, -- Symbol of token purchased
    token_sold_symbol,   -- Symbol of token sold
    token_bought_amount, -- Amount purchased (raw units)
    token_sold_amount,   -- Amount sold (raw units)
    amount_usd,          -- USD value of trade
    token_bought_address,-- Contract address of bought token
    token_sold_address,  -- Contract address of sold token
    taker,               -- Address executing the trade
    maker,               -- Address providing liquidity
    project_contract_address, -- DEX router/contract address
    tx_hash,             -- Transaction hash
    evt_index            -- Event index within transaction
FROM dex.trades
```

**Critical Performance Tips:**
- **ALWAYS filter by `blockchain`** when analyzing single chains
- **Use `block_date` for date ranges** (much faster than `block_time`)
- **Filter by `project`** to focus on specific DEXs
- **Use `amount_usd`** for volume analysis (pre-calculated)

#### 2. `dex_aggregator.trades` - Simplified Aggregator View

Shows high-level trades through DEX aggregators (1inch, 0x, etc.), complementing `dex.trades`.

**Key Difference:**
- One row in `dex_aggregator.trades` = User's intended trade
- Multiple rows in `dex.trades` = Individual steps/pools used

#### 3. `tokens.erc20` - Token Metadata and Prices

```sql
SELECT 
    blockchain,
    contract_address,
    symbol,
    decimals,           -- CRITICAL for amount calculations
    name,
    price_latest,       -- Most recent price
    price_latest_time   -- When price was last updated
FROM tokens.erc20
```

#### 4. `prices.usd` - Historical Price Data

```sql
SELECT 
    blockchain,
    contract_address,
    symbol,
    minute,             -- Minute-level precision
    price,              -- USD price at that minute
    decimals
FROM prices.usd
```

### **GOLDEN RULE #2: Always Join with Token Metadata**

```sql
-- ❌ WRONG - No token info, amounts are in raw units
SELECT 
    token_bought_amount,
    token_sold_amount
FROM dex.trades

-- ✅ CORRECT - Join with token metadata for readable amounts
SELECT 
    trades.token_bought_amount / power(10, bought_token.decimals) as bought_amount_formatted,
    trades.token_sold_amount / power(10, sold_token.decimals) as sold_amount_formatted,
    bought_token.symbol as bought_symbol,
    sold_token.symbol as sold_symbol
FROM dex.trades trades
LEFT JOIN tokens.erc20 bought_token 
    ON trades.token_bought_address = bought_token.contract_address 
    AND trades.blockchain = bought_token.blockchain
LEFT JOIN tokens.erc20 sold_token 
    ON trades.token_sold_address = sold_token.contract_address 
    AND trades.blockchain = sold_token.blockchain
```

---

## Smart Contract Data Analysis Fundamentals

### Understanding Contract Decoding

Smart contracts on Ethereum (and EVM chains) store their logic as bytecode. To make this human-readable, Dune uses **Application Binary Interfaces (ABIs)** to decode:

1. **Function calls** → `_call_` tables
2. **Events** → `_evt_` tables

### Decoded Table Naming Convention

```
[projectname_blockchain].[contractName]_[type]_[functionOrEventName]
```

**Examples:**
```sql
-- Uniswap V3 Pool swap events
uniswap_v3_ethereum.pool_evt_swap

-- Aave lending pool borrow function calls
aave_v2_ethereum.lendingpool_call_borrow

-- Compound cToken mint events
compound_v2_ethereum.cerc20_evt_mint
```

### **CRITICAL LIMITATION: State Data is NOT Available**

⚠️ **IMPORTANT:** Smart contract state data (storage variables) is **NOT** available on Dune.

**What's Available:**
- Function calls that resulted in transactions
- Events emitted during transactions
- External function calls between contracts

**What's NOT Available:**
- `view` or `pure` function calls made locally
- Contract storage variables
- Balances stored in contract state (unless emitted as events)

### Finding the Right Decoded Tables

#### Method 1: Use the Schema Browser
In Dune's interface, navigate to the schema browser and search for:
- Project name (e.g., "uniswap", "aave", "compound")
- Contract address
- Function or event name

#### Method 2: Query the contracts table
```sql
-- Find all decoded contracts for a project
SELECT DISTINCT 
    namespace,
    name,
    address
FROM ethereum.contracts 
WHERE namespace LIKE '%uniswap%'
```

#### Method 3: Search by contract address
```sql
-- Find decoded tables for a specific contract
SELECT DISTINCT 
    namespace,
    name
FROM ethereum.contracts 
WHERE address = 0x1f98431c8ad98523631ae4a59f267346ea31f984
```

### **GOLDEN RULE #3: Prefer Decoded Events Over Function Calls**

Events are more reliable and performant than function calls because:
- Events are indexed and searchable
- Events represent state changes
- Function calls may not complete successfully

```sql
-- ❌ SLOWER - Function call tables
SELECT * FROM uniswap_v3_ethereum.pool_call_swap

-- ✅ FASTER - Event tables  
SELECT * FROM uniswap_v3_ethereum.pool_evt_swap
```

---

## Query Performance Optimization Principles

### 1. Index-Aware Filtering

Dune's tables are indexed on specific columns. **Always filter on indexed columns first.**

**Primary Indexes:**
- `block_time` / `block_date` - Temporal filtering
- `block_number` - Block-based filtering  
- `blockchain` - Chain filtering
- `contract_address` - Contract-specific filtering

```sql
-- ✅ FAST - Uses indexes
SELECT * FROM dex.trades
WHERE blockchain = 'ethereum'           -- Indexed
  AND block_date >= '2024-01-01'        -- Indexed
  AND project = 'uniswap'               -- Potentially indexed
  AND amount_usd > 1000

-- ❌ SLOW - No index usage
SELECT * FROM dex.trades
WHERE token_bought_symbol = 'USDC'      -- Not indexed
  AND taker = 0x123...                  -- Not indexed
```

### 2. Date Range Optimization

**Use `block_date` for day-level analysis:**
```sql
-- ✅ FAST - Uses date partitioning
WHERE block_date >= '2024-01-01' 
  AND block_date < '2024-02-01'

-- ❌ SLOWER - Requires timestamp scanning
WHERE block_time >= '2024-01-01' 
  AND block_time < '2024-02-01'
```

**Use `block_time` only when you need hour/minute precision:**
```sql
-- ✅ ACCEPTABLE - When you need hourly data
WHERE block_time >= timestamp '2024-01-01 00:00:00'
  AND block_time < timestamp '2024-01-01 01:00:00'
```

### 3. Blockchain Filtering Strategy

**ALWAYS filter by blockchain early in cross-chain tables:**

```sql
-- ✅ FAST - Blockchain filter applied first
SELECT 
    project,
    SUM(amount_usd) as volume
FROM dex.trades
WHERE blockchain = 'ethereum'           -- Filter first!
  AND block_date >= '2024-01-01'
  AND project IN ('uniswap', 'sushiswap')
GROUP BY project

-- ❌ SLOW - No blockchain filter
SELECT 
    blockchain,
    project,
    SUM(amount_usd) as volume
FROM dex.trades
WHERE block_date >= '2024-01-01'
  AND project IN ('uniswap', 'sushiswap')  -- Scans all chains!
GROUP BY blockchain, project
```

### 4. Aggregation Optimization

**Use pre-aggregated columns when available:**

```sql
-- ✅ FAST - Uses pre-calculated USD amounts
SELECT 
    project,
    SUM(amount_usd) as total_volume_usd
FROM dex.trades
WHERE blockchain = 'ethereum'
GROUP BY project

-- ❌ SLOW - Manual price calculation
SELECT 
    project,
    SUM(token_bought_amount * p.price) as total_volume_usd
FROM dex.trades t
LEFT JOIN prices.usd p 
    ON t.token_bought_address = p.contract_address
    AND t.blockchain = p.blockchain
    AND date_trunc('minute', t.block_time) = p.minute
WHERE t.blockchain = 'ethereum'
GROUP BY project
```

### **GOLDEN RULE #4: Filter Early, Aggregate Late**

Structure your queries to apply the most selective filters first:

```sql
-- ✅ OPTIMAL QUERY STRUCTURE
WITH filtered_trades AS (
    SELECT *
    FROM dex.trades
    WHERE blockchain = 'ethereum'        -- Most selective filter first
      AND block_date >= '2024-01-01'    -- Time filter second
      AND project = 'uniswap'           -- Project filter third
      AND version = '3'                 -- Version filter fourth
      AND amount_usd > 1000              -- Amount filter last
)
SELECT 
    token_bought_symbol,
    token_sold_symbol,
    COUNT(*) as trade_count,
    SUM(amount_usd) as volume_usd
FROM filtered_trades
GROUP BY 1, 2
ORDER BY volume_usd DESC
```

---

## Advanced DeFi Analysis Patterns

### 1. Liquidity Pool Analysis

#### Finding Pool Addresses and Tracking Activity

```sql
-- Get top pools by trading volume
SELECT 
    pool_address,
    token_bought_symbol,
    token_sold_symbol,
    COUNT(*) as trade_count,
    SUM(amount_usd) as volume_usd,
    COUNT(DISTINCT taker) as unique_traders
FROM dex.trades
WHERE blockchain = 'ethereum'
  AND project = 'uniswap'
  AND version = '3'
  AND block_date >= current_date - interval '7' day
GROUP BY 1, 2, 3
HAVING SUM(amount_usd) > 100000
ORDER BY volume_usd DESC
```

#### Pool Performance Analysis

```sql
-- Analyze pool trading patterns and fees
WITH pool_stats AS (
    SELECT 
        pool_address,
        token_bought_symbol,
        token_sold_symbol,
        date_trunc('hour', block_time) as hour,
        COUNT(*) as trades_per_hour,
        SUM(amount_usd) as volume_per_hour,
        AVG(amount_usd) as avg_trade_size,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount_usd) as median_trade_size
    FROM dex.trades
    WHERE blockchain = 'ethereum'
      AND project = 'uniswap'
      AND version = '3'
      AND block_date >= current_date - interval '1' day
    GROUP BY 1, 2, 3, 4
)
SELECT 
    pool_address,
    token_bought_symbol || '/' || token_sold_symbol as pair,
    AVG(trades_per_hour) as avg_trades_per_hour,
    AVG(volume_per_hour) as avg_volume_per_hour,
    AVG(avg_trade_size) as avg_trade_size,
    AVG(median_trade_size) as avg_median_trade_size
FROM pool_stats
GROUP BY 1, 2
ORDER BY avg_volume_per_hour DESC
```

### 2. Cross-DEX Arbitrage Analysis

```sql
-- Find arbitrage opportunities across DEXs
WITH price_comparison AS (
    SELECT 
        block_time,
        token_bought_address,
        token_sold_address,
        project,
        amount_usd / token_bought_amount * POWER(10, 18) as price_per_token
    FROM dex.trades
    WHERE blockchain = 'ethereum'
      AND block_date = current_date
      AND token_bought_address = 0xa0b86a33e6a8e0b57c5e678b15dce2ffec3e9de5  -- USDC
      AND token_sold_address = 0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2   -- WETH
      AND amount_usd > 10000
),
price_ranges AS (
    SELECT 
        date_trunc('minute', block_time) as minute,
        project,
        MIN(price_per_token) as min_price,
        MAX(price_per_token) as max_price,
        COUNT(*) as trades
    FROM price_comparison
    GROUP BY 1, 2
    HAVING COUNT(*) >= 3  -- At least 3 trades per minute
)
SELECT 
    p1.minute,
    p1.project as dex_1,
    p2.project as dex_2,
    p1.min_price as price_dex_1,
    p2.max_price as price_dex_2,
    (p2.max_price - p1.min_price) / p1.min_price * 100 as arbitrage_opportunity_pct
FROM price_ranges p1
JOIN price_ranges p2 
    ON p1.minute = p2.minute 
    AND p1.project < p2.project  -- Avoid duplicate pairs
WHERE (p2.max_price - p1.min_price) / p1.min_price > 0.005  -- > 0.5% difference
ORDER BY arbitrage_opportunity_pct DESC
```

### 3. MEV and Front-Running Detection

```sql
-- Detect potential MEV transactions
WITH mev_analysis AS (
    SELECT 
        tx_hash,
        block_number,
        evt_index,
        project,
        amount_usd,
        token_bought_address,
        token_sold_address,
        taker,
        -- Look for transactions in same block
        COUNT(*) OVER (
            PARTITION BY block_number, token_bought_address, token_sold_address
        ) as trades_same_block_same_pair,
        -- Look for same taker multiple trades
        COUNT(*) OVER (
            PARTITION BY block_number, taker
        ) as trades_same_block_same_taker
    FROM dex.trades
    WHERE blockchain = 'ethereum'
      AND block_date >= current_date - interval '1' day
      AND amount_usd > 50000
)
SELECT 
    block_number,
    COUNT(*) as suspicious_trades,
    SUM(amount_usd) as total_volume,
    ARRAY_AGG(DISTINCT taker) as takers,
    ARRAY_AGG(DISTINCT project) as projects_used
FROM mev_analysis
WHERE trades_same_block_same_pair >= 3
   OR trades_same_block_same_taker >= 5
GROUP BY block_number
ORDER BY suspicious_trades DESC, total_volume DESC
```

### 4. Protocol TVL and Usage Analysis

```sql
-- Analyze protocol growth and usage patterns
WITH daily_metrics AS (
    SELECT 
        block_date,
        project,
        version,
        COUNT(*) as daily_trades,
        COUNT(DISTINCT taker) as daily_users,
        SUM(amount_usd) as daily_volume,
        AVG(amount_usd) as avg_trade_size
    FROM dex.trades
    WHERE blockchain = 'ethereum'
      AND block_date >= current_date - interval '30' day
    GROUP BY 1, 2, 3
),
growth_metrics AS (
    SELECT 
        *,
        LAG(daily_volume, 7) OVER (
            PARTITION BY project, version 
            ORDER BY block_date
        ) as volume_7d_ago,
        LAG(daily_users, 7) OVER (
            PARTITION BY project, version 
            ORDER BY block_date
        ) as users_7d_ago
    FROM daily_metrics
)
SELECT 
    project,
    version,
    SUM(daily_volume) as total_volume_30d,
    AVG(daily_users) as avg_daily_users,
    AVG(daily_trades) as avg_daily_trades,
    AVG(CASE 
        WHEN volume_7d_ago > 0 
        THEN (daily_volume - volume_7d_ago) / volume_7d_ago * 100 
    END) as avg_volume_growth_7d_pct,
    AVG(CASE 
        WHEN users_7d_ago > 0 
        THEN (daily_users - users_7d_ago) / users_7d_ago * 100 
    END) as avg_user_growth_7d_pct
FROM growth_metrics
WHERE block_date >= current_date - interval '7' day  -- Last 7 days for growth calc
GROUP BY 1, 2
ORDER BY total_volume_30d DESC
```

---

## Token Movement and Transfer Analysis

### Understanding Token Transfer Data

Token transfers are recorded as events in smart contracts. The most important tables:

1. **`erc20_ethereum.evt_transfer`** - All ERC20 transfers
2. **`tokens.transfers`** - Curated, cross-chain transfers  
3. **Individual token tables** - e.g., `usdc_ethereum.evt_transfer`

### **GOLDEN RULE #5: Use Specific Token Tables When Possible**

```sql
-- ✅ FASTEST - Specific token table
SELECT * FROM usdc_ethereum.evt_transfer
WHERE evt_block_time >= '2024-01-01'

-- ✅ FAST - Filtered general table  
SELECT * FROM erc20_ethereum.evt_transfer
WHERE contract_address = 0xa0b86a33e6a8e0b57c5e678b15dce2ffec3e9de5
  AND evt_block_time >= '2024-01-01'

-- ❌ SLOW - Unfiltered general table
SELECT * FROM erc20_ethereum.evt_transfer
WHERE evt_block_time >= '2024-01-01'
```

### 1. Large Transfer Detection

```sql
-- Find whale transactions and unusual movements
WITH large_transfers AS (
    SELECT 
        evt_block_time,
        evt_tx_hash,
        "from",
        "to",
        value / 1e6 as amount_usdc,  -- USDC has 6 decimals
        value
    FROM usdc_ethereum.evt_transfer
    WHERE evt_block_date >= current_date - interval '7' day
      AND value >= 1000000 * 1e6  -- $1M+ transfers
),
transfer_analysis AS (
    SELECT 
        *,
        -- Identify exchange addresses (simplified)
        CASE 
            WHEN "from" IN (
                0x503828976d22510aad0201ac7ec88293211d23da,  -- Coinbase
                0x28c6c06298d514db089934071355e5743bf21d60,  -- Binance
                0x21a31ee1afc51d94c2efccaa2092ad1028285549   -- Binance 2
            ) THEN 'Exchange Outflow'
            WHEN "to" IN (
                0x503828976d22510aad0201ac7ec88293211d23da,
                0x28c6c06298d514db089934071355e5743bf21d60,
                0x21a31ee1afc51d94c2efccaa2092ad1028285549
            ) THEN 'Exchange Inflow'
            ELSE 'Wallet to Wallet'
        END as transfer_type
    FROM large_transfers
)
SELECT 
    date_trunc('day', evt_block_time) as day,
    transfer_type,
    COUNT(*) as transfer_count,
    SUM(amount_usdc) as total_amount_usdc,
    AVG(amount_usdc) as avg_amount_usdc,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount_usdc) as median_amount_usdc
FROM transfer_analysis
GROUP BY 1, 2
ORDER BY day DESC, total_amount_usdc DESC
```

### 2. Token Holder Analysis

```sql
-- Analyze token distribution and holder behavior
WITH usdc_balances AS (
    SELECT 
        "to" as holder,
        SUM(value) as total_received
    FROM usdc_ethereum.evt_transfer
    WHERE evt_block_date >= '2024-01-01'
    GROUP BY "to"
    
    UNION ALL
    
    SELECT 
        "from" as holder,
        -SUM(value) as total_sent
    FROM usdc_ethereum.evt_transfer
    WHERE evt_block_date >= '2024-01-01'
      AND "from" != 0x0000000000000000000000000000000000000000  -- Exclude mints
    GROUP BY "from"
),
net_balances AS (
    SELECT 
        holder,
        SUM(total_received) / 1e6 as net_balance_usdc
    FROM usdc_balances
    GROUP BY holder
    HAVING SUM(total_received) > 0  -- Only positive balances
),
holder_categories AS (
    SELECT 
        *,
        CASE 
            WHEN net_balance_usdc >= 10000000 THEN 'Whale (>$10M)'
            WHEN net_balance_usdc >= 1000000 THEN 'Large ($1M-$10M)'
            WHEN net_balance_usdc >= 100000 THEN 'Medium ($100K-$1M)'
            WHEN net_balance_usdc >= 10000 THEN 'Small ($10K-$100K)'
            ELSE 'Retail (<$10K)'
        END as holder_category
    FROM net_balances
)
SELECT 
    holder_category,
    COUNT(*) as holder_count,
    SUM(net_balance_usdc) as total_balance_usdc,
    AVG(net_balance_usdc) as avg_balance_usdc,
    SUM(net_balance_usdc) / (SELECT SUM(net_balance_usdc) FROM holder_categories) * 100 as pct_of_total_supply
FROM holder_categories
GROUP BY holder_category
ORDER BY total_balance_usdc DESC
```

### 3. Cross-Chain Token Flow Analysis

```sql
-- Analyze token movements across chains using bridges
WITH bridge_addresses AS (
    SELECT address, bridge_name FROM (
        VALUES 
        (0x40ec5b33f54e0e8a33a975908c5ba1c14e5bbbdf, 'Polygon Bridge'),
        (0xa0c68c638235ee32657e8f720a23cec1bfc77c77, 'Polygon Bridge'),
        (0x8484ef722627bf18ca5ae6bcf031c23e6e922b30, 'Arbitrum Bridge'),
        (0xa3a7b6f88361f48403514059f1f16c8e78d60eec, 'Arbitrum Bridge')
    ) as t(address, bridge_name)
),
bridge_flows AS (
    SELECT 
        evt_block_date,
        ba.bridge_name,
        CASE 
            WHEN t."to" = ba.address THEN 'Inflow'
            WHEN t."from" = ba.address THEN 'Outflow'
        END as flow_direction,
        SUM(t.value / 1e6) as amount_usdc,
        COUNT(*) as transaction_count
    FROM usdc_ethereum.evt_transfer t
    JOIN bridge_addresses ba ON (t."to" = ba.address OR t."from" = ba.address)
    WHERE evt_block_date >= current_date - interval '30' day
    GROUP BY 1, 2, 3
)
SELECT 
    bridge_name,
    flow_direction,
    SUM(amount_usdc) as total_amount_30d,
    SUM(transaction_count) as total_transactions_30d,
    AVG(amount_usdc) as avg_daily_amount,
    AVG(transaction_count) as avg_daily_transactions
FROM bridge_flows
GROUP BY 1, 2
ORDER BY bridge_name, flow_direction
```

### 4. Token Velocity Analysis

```sql
-- Calculate token velocity and circulation patterns
WITH daily_transfers AS (
    SELECT 
        evt_block_date,
        COUNT(*) as transfer_count,
        SUM(value) as total_volume,
        COUNT(DISTINCT "from") as unique_senders,
        COUNT(DISTINCT "to") as unique_receivers,
        COUNT(DISTINCT CONCAT("from", "to")) as unique_pairs
    FROM usdc_ethereum.evt_transfer
    WHERE evt_block_date >= current_date - interval '30' day
      AND "from" != 0x0000000000000000000000000000000000000000  -- Exclude mints
      AND "to" != 0x0000000000000000000000000000000000000000    -- Exclude burns
    GROUP BY evt_block_date
),
velocity_metrics AS (
    SELECT 
        *,
        total_volume / 1e6 as daily_volume_usdc,
        total_volume / (SELECT SUM(value) FROM usdc_ethereum.evt_transfer 
                       WHERE evt_block_date = dt.evt_block_date 
                       AND "from" = 0x0000000000000000000000000000000000000000) * 100 as velocity_vs_mints
    FROM daily_transfers dt
)
SELECT 
    evt_block_date,
    transfer_count,
    daily_volume_usdc,
    unique_senders,
    unique_receivers,
    unique_pairs,
    ROUND(daily_volume_usdc / transfer_count, 2) as avg_transfer_size,
    ROUND(daily_volume_usdc / unique_pairs, 2) as avg_volume_per_pair,
    ROUND(velocity_vs_mints, 4) as velocity_pct
FROM velocity_metrics
ORDER BY evt_block_date DESC
```

---

## Common Performance Pitfalls and Solutions

### Pitfall 1: Using Raw Logs Instead of Decoded Tables

**❌ WRONG:**
```sql
-- This query will be extremely slow
SELECT * FROM ethereum.logs
WHERE contract_address = 0xa0b86a33e6a8e0b57c5e678b15dce2ffec3e9de5
  AND topic0 = 0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef
```

**✅ CORRECT:**
```sql
-- Use the decoded transfer events table
SELECT * FROM erc20_ethereum.evt_transfer
WHERE contract_address = 0xa0b86a33e6a8e0b57c5e678b15dce2ffec3e9de5
```

**Why it's faster:** Decoded tables are pre-processed, indexed, and human-readable.

### Pitfall 2: Not Filtering by Blockchain

**❌ WRONG:**
```sql
-- Scans ALL blockchains
SELECT project, SUM(amount_usd) FROM dex.trades
WHERE block_date >= '2024-01-01'
GROUP BY project
```

**✅ CORRECT:**
```sql
-- Only scans Ethereum data
SELECT project, SUM(amount_usd) FROM dex.trades
WHERE blockchain = 'ethereum'
  AND block_date >= '2024-01-01'
GROUP BY project
```

### Pitfall 3: Inefficient Date Filtering

**❌ WRONG:**
```sql
-- String comparison is slow
WHERE DATE(block_time) >= '2024-01-01'
```

**✅ CORRECT:**
```sql
-- Direct date comparison is fast
WHERE block_date >= '2024-01-01'
-- OR for timestamp precision:
WHERE block_time >= timestamp '2024-01-01 00:00:00'
```

### Pitfall 4: Unnecessary JOINs

**❌ WRONG:**
```sql
-- Joining when data is already available
SELECT 
    t.amount_usd,
    p.price
FROM dex.trades t
LEFT JOIN prices.usd p 
    ON t.token_bought_address = p.contract_address
WHERE t.blockchain = 'ethereum'
```

**✅ CORRECT:**
```sql
-- Use pre-calculated USD amounts
SELECT amount_usd
FROM dex.trades
WHERE blockchain = 'ethereum'
```

### Pitfall 5: Inefficient Aggregations

**❌ WRONG:**
```sql
-- Calculating prices manually
SELECT 
    AVG(token_bought_amount * price_usd) as avg_trade_value
FROM dex.trades
```

**✅ CORRECT:**
```sql
-- Use pre-calculated values
SELECT AVG(amount_usd) as avg_trade_value
FROM dex.trades
```

### Pitfall 6: Not Using LIMIT for Exploration

**❌ WRONG:**
```sql
-- Returns millions of rows
SELECT * FROM dex.trades
WHERE blockchain = 'ethereum'
```

**✅ CORRECT:**
```sql
-- Always limit when exploring
SELECT * FROM dex.trades
WHERE blockchain = 'ethereum'
LIMIT 1000
```

### Pitfall 7: Inefficient Window Functions

**❌ WRONG:**
```sql
-- Inefficient partitioning
SELECT 
    *,
    ROW_NUMBER() OVER (ORDER BY amount_usd DESC) as rn
FROM dex.trades
WHERE blockchain = 'ethereum'
```

**✅ CORRECT:**
```sql
-- Partition by relevant columns
SELECT 
    *,
    ROW_NUMBER() OVER (
        PARTITION BY project, block_date 
        ORDER BY amount_usd DESC
    ) as rn
FROM dex.trades
WHERE blockchain = 'ethereum'
  AND block_date >= '2024-01-01'
```

---

## Practical Query Examples

### Example 1: Daily DEX Volume Analysis

**Objective:** Analyze daily trading volume across different DEXs on Ethereum.

```sql
-- High-performance daily DEX volume analysis
WITH daily_dex_volume AS (
    SELECT 
        block_date,
        project,
        version,
        COUNT(*) as trade_count,
        COUNT(DISTINCT taker) as unique_traders,
        SUM(amount_usd) as volume_usd,
        AVG(amount_usd) as avg_trade_size_usd,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount_usd) as median_trade_size_usd
    FROM dex.trades
    WHERE blockchain = 'ethereum'                    -- Essential filter
      AND block_date >= current_date - interval '30' day  -- Time bound
      AND amount_usd > 0                            -- Valid trades only
    GROUP BY 1, 2, 3
),
market_share AS (
    SELECT 
        *,
        volume_usd / SUM(volume_usd) OVER (PARTITION BY block_date) * 100 as market_share_pct,
        LAG(volume_usd, 1) OVER (
            PARTITION BY project, version 
            ORDER BY block_date
        ) as prev_day_volume
    FROM daily_dex_volume
),
final_metrics AS (
    SELECT 
        *,
        CASE 
            WHEN prev_day_volume > 0 
            THEN (volume_usd - prev_day_volume) / prev_day_volume * 100 
        END as volume_change_pct
    FROM market_share
)
SELECT 
    block_date,
    project,
    version,
    trade_count,
    unique_traders,
    ROUND(volume_usd, 2) as volume_usd,
    ROUND(avg_trade_size_usd, 2) as avg_trade_size_usd,
    ROUND(median_trade_size_usd, 2) as median_trade_size_usd,
    ROUND(market_share_pct, 2) as market_share_pct,
    ROUND(volume_change_pct, 2) as volume_change_pct
FROM final_metrics
WHERE block_date >= current_date - interval '7' day  -- Last 7 days
ORDER BY block_date DESC, volume_usd DESC
```

### Example 2: Uniswap V3 Pool Performance Analysis

**Objective:** Analyze the performance of Uniswap V3 pools and identify the most active trading pairs.

```sql
-- Comprehensive Uniswap V3 pool analysis
WITH pool_trades AS (
    SELECT 
        pool_address,
        token_bought_address,
        token_sold_address,
        token_bought_symbol,
        token_sold_symbol,
        block_date,
        block_time,
        amount_usd,
        taker
    FROM dex.trades
    WHERE blockchain = 'ethereum'
      AND project = 'uniswap'
      AND version = '3'
      AND block_date >= current_date - interval '7' day
      AND amount_usd > 100  -- Filter small trades for performance
),
pool_metrics AS (
    SELECT 
        pool_address,
        token_bought_symbol,
        token_sold_symbol,
        COUNT(*) as total_trades,
        COUNT(DISTINCT taker) as unique_traders,
        SUM(amount_usd) as total_volume_usd,
        AVG(amount_usd) as avg_trade_size,
        STDDEV(amount_usd) as trade_size_stddev,
        MIN(amount_usd) as min_trade_size,
        MAX(amount_usd) as max_trade_size,
        COUNT(*) / 7.0 as avg_trades_per_day,
        SUM(amount_usd) / 7.0 as avg_volume_per_day
    FROM pool_trades
    GROUP BY 1, 2, 3
),
trading_hours AS (
    SELECT 
        pool_address,
        EXTRACT(hour FROM block_time) as hour_of_day,
        COUNT(*) as trades_in_hour,
        SUM(amount_usd) as volume_in_hour
    FROM pool_trades
    GROUP BY 1, 2
),
peak_hours AS (
    SELECT 
        pool_address,
        hour_of_day,
        trades_in_hour,
        volume_in_hour,
        ROW_NUMBER() OVER (
            PARTITION BY pool_address 
            ORDER BY volume_in_hour DESC
        ) as volume_rank
    FROM trading_hours
),
pool_peak_hour AS (
    SELECT 
        pool_address,
        hour_of_day as peak_volume_hour,
        volume_in_hour as peak_hour_volume
    FROM peak_hours
    WHERE volume_rank = 1
)
SELECT 
    pm.pool_address,
    pm.token_bought_symbol || '/' || pm.token_sold_symbol as trading_pair,
    pm.total_trades,
    pm.unique_traders,
    ROUND(pm.total_volume_usd, 2) as total_volume_usd,
    ROUND(pm.avg_trade_size, 2) as avg_trade_size,
    ROUND(pm.trade_size_stddev, 2) as trade_size_stddev,
    ROUND(pm.avg_trades_per_day, 1) as avg_trades_per_day,
    ROUND(pm.avg_volume_per_day, 2) as avg_volume_per_day,
    pph.peak_volume_hour,
    ROUND(pph.peak_hour_volume, 2) as peak_hour_volume,
    -- Calculate trader concentration
    ROUND(pm.total_volume_usd / pm.unique_traders, 2) as volume_per_trader,
    -- Trading efficiency metric
    ROUND(pm.total_trades::decimal / pm.total_volume_usd * 1000, 4) as trades_per_1k_volume
FROM pool_metrics pm
LEFT JOIN pool_peak_hour pph ON pm.pool_address = pph.pool_address
WHERE pm.total_volume_usd > 10000  -- Focus on active pools
ORDER BY pm.total_volume_usd DESC
LIMIT 50
```

### Example 3: Token Transfer Pattern Analysis

**Objective:** Analyze USDC transfer patterns to identify whale behavior and exchange flows.

```sql
-- Advanced USDC transfer pattern analysis
WITH transfer_data AS (
    SELECT 
        evt_block_time,
        evt_block_date,
        evt_tx_hash,
        "from",
        "to",
        value,
        value / 1e6 as amount_usdc,  -- USDC has 6 decimals
        evt_index
    FROM usdc_ethereum.evt_transfer
    WHERE evt_block_date >= current_date - interval '3' day
      AND value >= 100000 * 1e6  -- $100K+ transfers only
      AND "from" != 0x0000000000000000000000000000000000000000  -- Exclude mints
      AND "to" != 0x0000000000000000000000000000000000000000    -- Exclude burns
),
-- Identify known exchange and institutional addresses
address_labels AS (
    SELECT address, label FROM (
        VALUES 
        (0x503828976d22510aad0201ac7ec88293211d23da, 'Coinbase 1'),
        (0x28c6c06298d514db089934071355e5743bf21d60, 'Binance 1'),
        (0x21a31ee1afc51d94c2efccaa2092ad1028285549, 'Binance 2'),
        (0x47ac0fb4f2d84898e4d9e7b4dab3c24507a6d503, 'Binance 3'),
        (0x8eb8a3b98659cce290402893d0123abb75e3ab28, 'FTX US'),
        (0x2910543af39aba0cd09dbb2d50200b3e800a63d2, 'Kraken 1'),
        (0x267be1c1d684f78cb4f6a176c4911b741e4ffdc0, 'Kraken 2'),
        (0x1522900b6dafac587d499a862861c0869be6e428, 'Grayscale'),
        (0x3cc936b795a188f0e246cbb2d74c5bd190aecf18, 'Cumberland DRW'),
        (0x5041ed759dd4afc3a72b8192c143f72f4724081a, 'Circle (USDC)'),
        (0x5b76f5b8fc9d700624f78315c9966b45d61d3f71, 'Alameda Research'),
        (0x477573f212a7bdd5f7c12889bd1ad0aa44fb82aa, 'Wintermute')
    ) as t(address, label)
),
labeled_transfers AS (
    SELECT 
        td.*,
        COALESCE(al_from.label, 'Unknown') as from_label,
        COALESCE(al_to.label, 'Unknown') as to_label,
        CASE 
            WHEN al_from.label IS NOT NULL AND al_to.label IS NULL THEN 'Exchange Outflow'
            WHEN al_from.label IS NULL AND al_to.label IS NOT NULL THEN 'Exchange Inflow'
            WHEN al_from.label IS NOT NULL AND al_to.label IS NOT NULL THEN 'Exchange to Exchange'
            ELSE 'Wallet to Wallet'
        END as transfer_type
    FROM transfer_data td
    LEFT JOIN address_labels al_from ON td."from" = al_from.address
    LEFT JOIN address_labels al_to ON td."to" = al_to.address
),
-- Analyze transaction clustering (potential related transfers)
tx_analysis AS (
    SELECT 
        evt_tx_hash,
        COUNT(*) as transfers_in_tx,
        SUM(amount_usdc) as total_amount_in_tx,
        ARRAY_AGG(DISTINCT from_label) as from_entities,
        ARRAY_AGG(DISTINCT to_label) as to_entities,
        ARRAY_AGG(DISTINCT transfer_type) as transfer_types
    FROM labeled_transfers
    GROUP BY evt_tx_hash
),
-- Daily summary by transfer type
daily_summary AS (
    SELECT 
        evt_block_date,
        transfer_type,
        COUNT(*) as transfer_count,
        SUM(amount_usdc) as total_amount,
        AVG(amount_usdc) as avg_amount,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount_usdc) as median_amount,
        COUNT(DISTINCT "from") as unique_senders,
        COUNT(DISTINCT "to") as unique_receivers
    FROM labeled_transfers
    GROUP BY 1, 2
),
-- Whale activity analysis
whale_analysis AS (
    SELECT 
        evt_block_date,
        "from",
        "to",
        from_label,
        to_label,
        COUNT(*) as transfer_count,
        SUM(amount_usdc) as total_moved,
        MAX(amount_usdc) as largest_transfer,
        AVG(amount_usdc) as avg_transfer_size
    FROM labeled_transfers
    WHERE amount_usdc >= 1000000  -- $1M+ whale transactions
    GROUP BY 1, 2, 3, 4, 5
    HAVING COUNT(*) >= 2  -- Multiple large transfers
)
-- Final output combining multiple analyses
SELECT 
    'Daily Summary' as analysis_type,
    evt_block_date::varchar as date_or_entity,
    transfer_type as category,
    transfer_count::varchar as metric_1,
    ROUND(total_amount, 0)::varchar as metric_2,
    ROUND(avg_amount, 0)::varchar as metric_3,
    unique_senders::varchar as metric_4
FROM daily_summary
WHERE evt_block_date >= current_date - interval '3' day

UNION ALL

SELECT 
    'Whale Activity' as analysis_type,
    from_label || ' -> ' || to_label as date_or_entity,
    'Large Transfers' as category,
    transfer_count::varchar as metric_1,
    ROUND(total_moved, 0)::varchar as metric_2,
    ROUND(largest_transfer, 0)::varchar as metric_3,
    ROUND(avg_transfer_size, 0)::varchar as metric_4
FROM whale_analysis
WHERE evt_block_date >= current_date - interval '1' day

ORDER BY analysis_type, metric_2 DESC
```

### Example 4: Cross-Protocol Yield Farming Analysis

**Objective:** Analyze yield farming behavior across different DeFi protocols by tracking token movements and DEX interactions.

```sql
-- Cross-protocol yield farming behavior analysis
WITH farming_tokens AS (
    -- Define major farming/governance tokens
    SELECT token_address, symbol FROM (
        VALUES 
        (0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9, 'AAVE'),
        (0x6b3595068778dd592e39a122f4f5a5cf09c90fe2, 'SUSHI'),
        (0x1f9840a85d5af5bf1d1762f925bdaddc4201f984, 'UNI'),
        (0x514910771af9ca656af840dff83e8264ecf986ca, 'LINK'),
        (0xd33526068d116ce69f19a9ee46f0bd304f21a51f, 'RPL'),
        (0x6de037ef9ad2725eb40118bb1702ebb27e4aeb24, 'RNDR')
    ) as t(token_address, symbol)
),
-- Get recent farming token trades
farming_trades AS (
    SELECT 
        dt.block_date,
        dt.block_time,
        dt.tx_hash,
        dt.taker,
        dt.project,
        dt.version,
        ft.symbol,
        dt.token_bought_address,
        dt.token_sold_address,
        dt.token_bought_symbol,
        dt.token_sold_symbol,
        dt.amount_usd,
        CASE 
            WHEN dt.token_bought_address = ft.token_address THEN 'Buy'
            WHEN dt.token_sold_address = ft.token_address THEN 'Sell'
        END as action_type
    FROM dex.trades dt
    JOIN farming_tokens ft ON (
        dt.token_bought_address = ft.token_address 
        OR dt.token_sold_address = ft.token_address
    )
    WHERE dt.blockchain = 'ethereum'
      AND dt.block_date >= current_date - interval '7' day
      AND dt.amount_usd > 1000  -- Significant trades only
),
-- Identify yield farmers (users trading multiple farming tokens)
farmer_identification AS (
    SELECT 
        taker,
        COUNT(DISTINCT symbol) as tokens_traded,
        COUNT(DISTINCT project) as dexs_used,
        COUNT(*) as total_trades,
        SUM(amount_usd) as total_volume,
        SUM(CASE WHEN action_type = 'Buy' THEN amount_usd ELSE 0 END) as buy_volume,
        SUM(CASE WHEN action_type = 'Sell' THEN amount_usd ELSE 0 END) as sell_volume
    FROM farming_trades
    GROUP BY taker
    HAVING COUNT(DISTINCT symbol) >= 2  -- Traded at least 2 different farming tokens
       AND COUNT(*) >= 5                -- At least 5 trades
),
-- Analyze farming patterns by token
token_farming_patterns AS (
    SELECT 
        ft.symbol,
        COUNT(DISTINCT ft.taker) as unique_farmers,
        COUNT(*) as total_trades,
        SUM(ft.amount_usd) as total_volume,
        AVG(ft.amount_usd) as avg_trade_size,
        SUM(CASE WHEN ft.action_type = 'Buy' THEN ft.amount_usd ELSE 0 END) as total_buy_volume,
        SUM(CASE WHEN ft.action_type = 'Sell' THEN ft.amount_usd ELSE 0 END) as total_sell_volume,
        COUNT(DISTINCT ft.project) as dexs_used_for_token
    FROM farming_trades ft
    JOIN farmer_identification fi ON ft.taker = fi.taker
    GROUP BY ft.symbol
),
-- Cross-protocol behavior analysis
cross_protocol_behavior AS (
    SELECT 
        fi.taker,
        fi.tokens_traded,
        fi.dexs_used,
        fi.total_trades,
        fi.total_volume,
        (fi.buy_volume - fi.sell_volume) as net_position_change,
        CASE 
            WHEN fi.buy_volume > fi.sell_volume * 1.1 THEN 'Net Buyer'
            WHEN fi.sell_volume > fi.buy_volume * 1.1 THEN 'Net Seller'
            ELSE 'Balanced Trader'
        END as trader_type,
        -- Calculate farming score based on diversity and volume
        (fi.tokens_traded * fi.dexs_used * log(fi.total_volume)) as farming_score
    FROM farmer_identification fi
),
-- Top farmers analysis
top_farmers AS (
    SELECT 
        taker,
        tokens_traded,
        dexs_used,
        total_trades,
        total_volume,
        trader_type,
        farming_score,
        ROW_NUMBER() OVER (ORDER BY farming_score DESC) as farmer_rank
    FROM cross_protocol_behavior
)
-- Final comprehensive output
SELECT 
    'Token Analysis' as category,
    symbol as detail,
    unique_farmers::varchar as farmers,
    total_trades::varchar as trades,
    ROUND(total_volume, 0)::varchar as volume_usd,
    ROUND(total_buy_volume, 0)::varchar as buy_volume,
    ROUND(total_sell_volume, 0)::varchar as sell_volume,
    dexs_used_for_token::varchar as dex_count
FROM token_farming_patterns

UNION ALL

SELECT 
    'Top Farmers' as category,
    'Rank ' || farmer_rank::varchar as detail,
    tokens_traded::varchar as farmers,
    total_trades::varchar as trades,
    ROUND(total_volume, 0)::varchar as volume_usd,
    trader_type as buy_volume,
    ROUND(farming_score, 2)::varchar as sell_volume,
    dexs_used::varchar as dex_count
FROM top_farmers
WHERE farmer_rank <= 10

ORDER BY category DESC, volume_usd::numeric DESC
```

---

## Advanced Optimization Techniques

### 1. Using Materialized Views for Complex Calculations

For frequently-used complex queries, create materialized views to cache results:

```sql
-- Example: Create a materialized view for daily DEX metrics
CREATE MATERIALIZED VIEW daily_dex_metrics AS
WITH daily_stats AS (
    SELECT 
        blockchain,
        project,
        version,
        block_date,
        COUNT(*) as trade_count,
        COUNT(DISTINCT taker) as unique_traders,
        SUM(amount_usd) as volume_usd,
        AVG(amount_usd) as avg_trade_size,
        COUNT(DISTINCT token_bought_address) as unique_tokens_bought,
        COUNT(DISTINCT token_sold_address) as unique_tokens_sold
    FROM dex.trades
    WHERE block_date >= '2024-01-01'
    GROUP BY 1, 2, 3, 4
)
SELECT 
    *,
    volume_usd / SUM(volume_usd) OVER (PARTITION BY blockchain, block_date) * 100 as market_share_pct,
    LAG(volume_usd, 1) OVER (
        PARTITION BY blockchain, project, version 
        ORDER BY block_date
    ) as prev_day_volume
FROM daily_stats;

-- Use the materialized view for fast analysis
SELECT 
    project,
    AVG(market_share_pct) as avg_market_share,
    AVG(volume_usd) as avg_daily_volume
FROM daily_dex_metrics
WHERE blockchain = 'ethereum'
  AND block_date >= current_date - interval '30' day
GROUP BY project
ORDER BY avg_daily_volume DESC;
```

### 2. Optimizing Subqueries with CTEs

Replace correlated subqueries with CTEs for better performance:

```sql
-- ❌ SLOW - Correlated subqueries
SELECT 
    taker,
    amount_usd,
    (SELECT AVG(amount_usd) FROM dex.trades dt2 
     WHERE dt2.taker = dt1.taker) as user_avg_trade_size
FROM dex.trades dt1
WHERE blockchain = 'ethereum';

-- ✅ FAST - Using CTEs
WITH user_averages AS (
    SELECT 
        taker,
        AVG(amount_usd) as avg_trade_size
    FROM dex.trades
    WHERE blockchain = 'ethereum'
    GROUP BY taker
)
SELECT 
    dt.taker,
    dt.amount_usd,
    ua.avg_trade_size as user_avg_trade_size
FROM dex.trades dt
JOIN user_averages ua ON dt.taker = ua.taker
WHERE dt.blockchain = 'ethereum';
```

### 3. Efficient Pagination for Large Result Sets

When dealing with large result sets, use efficient pagination:

```sql
-- ✅ EFFICIENT PAGINATION
WITH ranked_trades AS (
    SELECT 
        *,
        ROW_NUMBER() OVER (ORDER BY amount_usd DESC, tx_hash) as rn
    FROM dex.trades
    WHERE blockchain = 'ethereum'
      AND block_date >= current_date - interval '1' day
)
SELECT *
FROM ranked_trades
WHERE rn BETWEEN 1001 AND 2000  -- Page 2 (1000 records per page)
ORDER BY rn;
```

### 4. Using UNION ALL for Conditional Logic

Instead of complex CASE statements, use UNION ALL for better performance:

```sql
-- ❌ SLOWER - Complex CASE statement
SELECT 
    taker,
    CASE 
        WHEN amount_usd > 100000 THEN 'Whale'
        WHEN amount_usd > 10000 THEN 'Large'
        ELSE 'Regular'
    END as trader_type,
    COUNT(*) as trade_count
FROM dex.trades
WHERE blockchain = 'ethereum'
GROUP BY 1, 2;

-- ✅ FASTER - UNION ALL approach
SELECT 'Whale' as trader_type, taker, COUNT(*) as trade_count
FROM dex.trades
WHERE blockchain = 'ethereum' AND amount_usd > 100000
GROUP BY taker

UNION ALL

SELECT 'Large' as trader_type, taker, COUNT(*) as trade_count
FROM dex.trades
WHERE blockchain = 'ethereum' AND amount_usd BETWEEN 10000 AND 100000
GROUP BY taker

UNION ALL

SELECT 'Regular' as trader_type, taker, COUNT(*) as trade_count
FROM dex.trades
WHERE blockchain = 'ethereum' AND amount_usd < 10000
GROUP BY taker;
```

### 5. Leveraging Array Functions for Efficient Analysis

Use array aggregation for complex grouping operations:

```sql
-- Efficient multi-dimensional analysis using arrays
WITH trader_activity AS (
    SELECT 
        taker,
        ARRAY_AGG(DISTINCT project) as projects_used,
        ARRAY_AGG(DISTINCT token_bought_symbol) as tokens_bought,
        ARRAY_AGG(DISTINCT token_sold_symbol) as tokens_sold,
        COUNT(*) as total_trades,
        SUM(amount_usd) as total_volume
    FROM dex.trades
    WHERE blockchain = 'ethereum'
      AND block_date >= current_date - interval '30' day
    GROUP BY taker
)
SELECT 
    taker,
    CARDINALITY(projects_used) as unique_projects,
    CARDINALITY(tokens_bought) as unique_tokens_bought,
    CARDINALITY(tokens_sold) as unique_tokens_sold,
    total_trades,
    total_volume,
    -- Calculate diversity score
    (CARDINALITY(projects_used) * CARDINALITY(tokens_bought)) as diversity_score
FROM trader_activity
WHERE CARDINALITY(projects_used) >= 3  -- Active on multiple DEXs
ORDER BY diversity_score DESC;
```

---

## Troubleshooting and Debugging

### 1. Query Performance Diagnosis

#### Check Query Execution Statistics
After running a query, always check the execution statistics:
- Click "last run X seconds ago" in the query editor
- Look for high CPU time, memory usage, or rows scanned
- Identify bottlenecks in your query plan

#### Use EXPLAIN for Complex Queries
```sql
EXPLAIN (ANALYZE, BUFFERS)
SELECT 
    project,
    SUM(amount_usd) as volume
FROM dex.trades
WHERE blockchain = 'ethereum'
  AND block_date >= '2024-01-01'
GROUP BY project;
```

### 2. Common Error Messages and Solutions

#### "Query timeout" Errors
**Cause:** Query taking longer than 30 minutes to execute.
**Solutions:**
- Add more selective filters (blockchain, date range)
- Use LIMIT for exploration
- Break complex queries into smaller parts
- Use materialized views for expensive calculations

#### "Out of memory" Errors
**Cause:** Query processing too much data in memory.
**Solutions:**
- Reduce data volume with filters
- Avoid large JOINs without proper filtering
- Use streaming aggregation instead of window functions where possible

#### "Rate limit exceeded" Errors
**Cause:** Too many API calls or query executions.
**Solutions:**
- Space out your query executions
- Use scheduled queries instead of manual execution
- Optimize queries to run faster and use fewer resources

### 3. Data Quality Validation

#### Validate Token Decimals
```sql
-- Check if token decimals are correctly applied
SELECT 
    token_address,
    symbol,
    decimals,
    MIN(amount) as min_raw_amount,
    MAX(amount) as max_raw_amount,
    MIN(amount / POWER(10, decimals)) as min_formatted,
    MAX(amount / POWER(10, decimals)) as max_formatted
FROM erc20_ethereum.evt_transfer
JOIN tokens.erc20 USING (contract_address, blockchain)
WHERE evt_block_date >= current_date - interval '1' day
GROUP BY 1, 2, 3
HAVING MAX(amount / POWER(10, decimals)) > 1e12  -- Suspiciously large amounts
```

#### Check for Missing Data
```sql
-- Identify gaps in daily data
WITH date_series AS (
    SELECT generate_series(
        date '2024-01-01',
        current_date,
        interval '1 day'
    )::date as check_date
),
daily_counts AS (
    SELECT 
        block_date,
        COUNT(*) as trade_count
    FROM dex.trades
    WHERE blockchain = 'ethereum'
      AND project = 'uniswap'
    GROUP BY block_date
)
SELECT 
    ds.check_date,
    COALESCE(dc.trade_count, 0) as trades,
    CASE 
        WHEN dc.trade_count IS NULL THEN 'Missing Data'
        WHEN dc.trade_count < 1000 THEN 'Low Activity'
        ELSE 'Normal'
    END as status
FROM date_series ds
LEFT JOIN daily_counts dc ON ds.check_date = dc.block_date
WHERE ds.check_date >= '2024-01-01'
ORDER BY ds.check_date DESC;
```

### 4. Testing and Validation Strategies

#### Sample Data Testing
Always test your queries on small datasets first:

```sql
-- Test your logic on a small sample
WITH sample_data AS (
    SELECT * FROM dex.trades
    WHERE blockchain = 'ethereum'
      AND block_date = '2024-01-15'  -- Single day
    LIMIT 10000
)
-- Your complex query logic here using sample_data
SELECT COUNT(*) FROM sample_data;
```

#### Cross-Validation with Known Results
Validate your results against known benchmarks:

```sql
-- Cross-validate DEX volume with external sources
SELECT 
    block_date,
    project,
    SUM(amount_usd) as dune_volume
FROM dex.trades
WHERE blockchain = 'ethereum'
  AND project = 'uniswap'
  AND version = '3'
  AND block_date = '2024-01-15'  -- Known high-volume day
GROUP BY 1, 2;
-- Compare with external DEX analytics platforms
```

---

## Final Best Practices Summary

### The 10 Commandments of High-Performance DuneSQL

1. **Always filter by blockchain first** in cross-chain tables
2. **Use curated tables** (`dex.trades`, `tokens.erc20`) over raw data
3. **Filter by date ranges** using `block_date` for performance
4. **Leverage decoded event tables** instead of raw logs
5. **Apply the most selective filters first** in WHERE clauses
6. **Use LIMIT during query development** and testing
7. **Join on indexed columns** (addresses, block numbers, timestamps)
8. **Avoid unnecessary JOINs** when data is already available
9. **Test on small datasets** before running on full data
10. **Monitor query execution statistics** to identify bottlenecks

### Query Development Workflow

1. **Start Small:** Begin with LIMIT 1000 and basic filters
2. **Add Filters:** Progressively add blockchain, date, and project filters
3. **Build Logic:** Implement your analysis logic on the filtered dataset
4. **Optimize:** Remove LIMIT and optimize for full dataset
5. **Validate:** Cross-check results with known benchmarks
6. **Document:** Comment your query logic for future reference

### Performance Checklist

Before running any production query, verify:

- [ ] Blockchain filter applied to cross-chain tables
- [ ] Date range specified using `block_date` or `block_time`
- [ ] Most selective filters applied first
- [ ] Using highest-level data tables available
- [ ] JOINs are necessary and properly indexed
- [ ] No correlated subqueries (use CTEs instead)
- [ ] LIMIT clause during development phase
- [ ] Query execution time is reasonable (<5 minutes)

---

This guide represents the comprehensive knowledge you need to write high-performance DuneSQL queries for DeFi protocol analysis. Master these concepts, and you'll be able to analyze any DeFi protocol, track token movements, and uncover insights from blockchain data efficiently and effectively.

Remember: **Performance is not just about speed—it's about getting reliable, accurate results that help you make better decisions in the fast-moving world of DeFi.**