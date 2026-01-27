# Lovdata API – Public Data Documentation Summary

**Extracted and formatted from provided Lovdata API information**  
**Last known reference**: Based on Lovdata's public documentation structure (circa 2025–2026)  
**Purpose**: Quick reference for developers using the free, no-authentication public datasets

## Introduction

Lovdata provides **two primary free datasets** containing current (gjeldende) Norwegian laws and central regulations. These datasets require **no user account, no authentication, and no API key**.

They are directly downloadable as compressed archives and licensed under **NLOD 2.0** (Norsk lisens for offentlige data 2.0), which permits broad reuse including personal, research, and commercial purposes with attribution.

### Direct Download Links

- **Current Laws (gjeldende lover)**  
  https://api.lovdata.no/v1/publicData/get/gjeldende-lover.tar.bz2

- **Current Central Regulations (gjeldende sentrale forskrifter)**  
  https://api.lovdata.no/v1/publicData/get/gjeldende-sentrale-forskrifter.tar.bz2

- **Additional free public datasets**  
  Browse the full list and other available files:  
  https://api.lovdata.no/swagger#/Public%20data

## General Information

The **Lovdata API** offers various endpoints for accessing document content, metadata, search, and other functions.

- Most endpoints **require authentication** via an API account provided by Lovdata (with the "api" role).
- The **Public data endpoints** (listed above) are the **main exceptions** — they are completely open and require no credentials.

The API is actively maintained:
- New methods and fields are added regularly.
- Breaking changes are introduced only in new API versions with distinct base URLs.

## Document Identifiers

Lovdata uses identifiers based on the **FRBR** model (Functional Requirements for Bibliographic Records):

- **refID** — corresponds to the FRBR **"work"** level  
  Represents the abstract intellectual entity (e.g., the Penal Code as a concept).

- **dokID** — corresponds to the FRBR **"expression"** level  
  Represents a specific manifestation/version of the work.

### Example: Straffeloven (Penal Code)

All of the following share the same **refID**:

**refID**: `lov/2005-05-20-28`

Corresponding **dokID** examples:

- Originally promulgated statute in Lovtidend:  
  **dokID**: `LTI/lov/2005-05-20-28`

- Current consolidated act (gjeldende versjon):  
  **dokID**: `NL/lov/2005-05-20-28`

- English translation/version:  
  **dokID**: `NLE/lov/2005-05-20-28`

- Revoked/historical version (if a new Penal Code replaces it in the future):  
  **dokID**: `NLO/lov/2005-05-20-28` (example — does not yet exist)

This structure allows multiple versions, languages, and publication formats of the same law to be clearly distinguished while remaining linked under one conceptual work (refID).

## Key Takeaways for Developers

- For **prototyping, personal projects, offline tools, or low-volume private use** → use the free bulk downloads (`gjeldende-lover.tar.bz2` and `gjeldende-sentrale-forskrifter.tar.bz2`).
- These contain the **current consolidated versions** of all laws and central regulations in structured, machine-readable format (primarily XML/HTML-like).
- For **real-time search, metadata queries, historical versions, or high-frequency access** → the authenticated REST API is required (commercial/subscription basis).
- Always include **attribution** to Lovdata and reference **NLOD 2.0** when redistributing or building products.

**Official source reference**: Lovdata API documentation & public data section[](https://api.lovdata.no/swagger)

*Document formatted for clarity – January 2026*
