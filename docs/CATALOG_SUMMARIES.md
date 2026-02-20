# Law Catalog Summary Status

## Overview

This document tracks the status of law summaries in the Lovli catalogs.

**Note:** The data files are stored in Google Drive and are not committed to git (see `.gitignore`).

| Catalog | Total Laws | With Summaries | Notes |
|---------|------------|----------------|-------|
| `law_catalog.json` | 4427 | 4353 (98%) | Merged catalog (sf + nl) |
| `law_catalog_nl.json` | 771 | 697 (90%) | Norwegian laws (nl prefix) |
| `law_catalog_sf.json` | 3656 | 3656 (100%) | Swiss regulations (sf prefix) |

## Missing Summaries

The following nl laws do not have summaries (74 total):

```
nl-18450607-000
nl-18880623-003
nl-19250807-000
nl-19270701-001
nl-19300227-002
nl-19300314-000
nl-19320527-002
nl-19390217-002
nl-19461011-001
nl-19470426-001
nl-19490728-026
nl-19501215-007
nl-19510629-034
nl-19510706-004
nl-19521121-003
nl-19530529-003
nl-19530626-011
nl-19530703-002
nl-19540301-000
nl-19550916-001
nl-19570424-001
nl-19570424-002
nl-19570921-002
nl-19610610-001
nl-19610610-002
nl-19610610-003
nl-19610610-004
nl-19610610-005
nl-19610610-006
nl-19610610-007
nl-19610610-008
nl-19610610-009
nl-19610610-010
nl-19610610-011
nl-19610610-012
nl-19610610-013
nl-19610610-014
nl-19610610-015
nl-19610610-016
nl-19610610-017
nl-19610610-018
nl-19610610-019
nl-19610610-020
nl-19610610-021
nl-19610610-022
nl-19610610-023
nl-19610610-024
nl-19610610-025
nl-19610610-026
nl-19610610-027
nl-19610610-028
nl-19610610-029
nl-19610610-030
nl-19610610-031
nl-19610610-032
nl-19610610-033
nl-19610610-034
nl-19610610-035
nl-19610610-036
nl-19610610-037
nl-19610610-038
nl-19610610-039
nl-19610610-040
nl-19610610-041
nl-19610610-042
nl-19610610-043
nl-19610610-044
nl-19610610-045
nl-19610610-046
nl-19610610-047
nl-19610610-048
nl-19610610-049
nl-19610610-050
```

## How to Generate Missing Summaries

To backfill the missing summaries, run:

```bash
export LANGSMITH_API_KEY=""
export LANGCHAIN_API_KEY=""
export LANGCHAIN_TRACING_V2=false
python scripts/build_catalog.py data/nl/ --output data/law_catalog_nl.json --backfill --concurrency 3
```

Then merge back into the main catalog:

```bash
python -c "
import json
with open('data/law_catalog.json') as f:
    sf = json.load(f)
with open('data/law_catalog_nl.json') as f:
    nl = json.load(f)
seen = {e['law_id']: e for e in sf}
seen.update({e['law_id']: e for e in nl})
with open('data/law_catalog.json', 'w') as f:
    json.dump(list(seen.values()), f, ensure_ascii=False, indent=2)
"
```

## File Descriptions

- `law_catalog.json` - Main merged catalog (used by the system)
- `law_catalog_nl.json` - Norwegian laws (nl prefix)
- `law_catalog_sf.json` - Swiss regulations (sf prefix) - backup
- `nl_missing_summaries.txt` - List of nl laws missing summaries
