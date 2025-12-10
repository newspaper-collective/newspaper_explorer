## Current Priorities
- update requirements.txt and requirements-dev.txt

- textblock coords?

+ Bohnſtedt (Graudenz) des Danziger Inſ.⸗Regts. Nr. 128, -> wrong long s test


- add source_name to aggregated blocks

- yolo, spacy, hf cache dir move to .env


- do bertopic year by year

- normalization output always named textblocks

### UI Issues
- use image index consitently
- datamanager tab?
- permalink to pages in issue browser to jump to them from other tabs
- page samples not reactive slightly buggy
- add drop shadows all around to flat elements
- support preprocessing metadata


### Data Issues
- Images need fixing too
- 1900 issue mets file is needded

- **Data integrity: Mixed issues in same directory**
  - 12 directories have ALTO files from multiple issues mixed together (0.09% of data)
  - Example: `1901/03/19/01` has issue 103 METS but contains ALTO files from both 103 and 104
  - Issue 103: Missing pages 13-20 (expected by METS, not found)
  - Issue 104: Has pages 1-8 in wrong directory with no METS file
  - Images are also affected/mixed up
  - Need to investigate if files are mislabeled or if data is genuinely incomplete
  - Validation command exists: `newspaper-explorer data validate-alto-mets --source der_tag`
  - Detection in fixes.py exists but no automatic repair (returns 0)
  - **Action**: Manual investigation of source data or contact data provider

- Finish downloading and fixing dataset -> Final testing

### Features
- CLI option to cleanup downloads



]Skipping 3074409X_1901-07-14_000_297_H_1_-01.xml: Missing required ID components (date=None, issue=None, daily=None, page=None)
march 1901


Transkribus OCR -> Problem with alto-export

- validate xml files


- combine results with/in knowdledge graph

- keywords ui broken


- sort dataset by filename when loading

- jsonrepair libary for llm client


- null values in image_index.parquet

- option to collapse analysis header


pagination for entiteies list

refactor captions cli

picture detail dialog should link to page.

thumbnail gallery lazy loading broken

- emotions: link to pages in noticable peaks


page reader -> click line scroll text


sorting broken in search -> 1910 Daten problem?

brwose link layout detection detailed region in analysis tab to overlay

issuereader, link emotion detections to line

simple stats page? word count etc?

drowpdon filters in browse month view


support external image urls -> include in index


Issue reader:
    1901
    >
    January
    >
    8.1.1901
    >
    Page 3
        -> Textline sorting broken, data or ui issue?