# Verified Generator calibration (Generator Part B, Half 2 -- G2)

First real-data run of B4 completeness and the B5 full chain against a real Granite generator and the TRUE verifier (docs/generator/verifier-backends.md). Measurement only; no thresholds or configs committed here.

**Hold-out respected:** HotpotQA, RGB and MuSiQue-Full are never loaded. Calibration data only -- ALCE/ASQA.

Config: seed 13, 60 cases requested, top-5 selected evidence, NLI backend `true`, Granite `ibm-granite/granite-4.1-3b`.

## Co-residency (Task 2) -- proven, not assumed

The B5 chain re-invokes Granite (completeness, evidence-recheck) *after* the TRUE
attribution pass, so the two models genuinely interleave within one query and
cannot be split into generate-then-verify stages without reimplementing the
orchestrator. Decision: hold both resident and measure. From the job log:

```
[mem] after first B5 query (Granite + TRUE resident): peak allocated 27.8GB / reserved 27.9GB of 39.2GB on NVIDIA A100-SXM4-80GB MIG 3g.40gb
[mem] after full B5 run:                              peak allocated 27.8GB / reserved 27.9GB of 39.2GB
```

Note the scheduler placed the job on a **3g.40gb MIG slice (39.2GB)**, not a full
a100: Granite-4.1-3b + TRUE co-reside at **27.8GB peak**, ~11GB under even the
40GB slice. So co-residency is comfortable here -- the earlier double-load OOM was
a property of the models loaded then, not of holding two models per se. The
run_verified_generator.slurm comment asking for a full a100 is over-cautious; the
40GB slice suffices.

## B4 -- completeness (Granite only)

| metric | value | reading |
|---|---|---|
| own facts read as covered | 92/117 (0.786) | gold answer states them, so higher is better (coverage recall) |
| foreign facts read as uncovered | 60/60 (1.000) | answer does NOT state them, so higher is better (coverage precision) |
| gap questions that are generic | 4/60 (0.067) | lower is better -- generic questions make evidence_recheck useless |
| constraint (year) survives into gap question | 11/26 (0.423) | of gaps whose case carried a year constraint |

### Verbatim gap questions (foreign-fact trials)

- fact `De revolutionibus orbium coelestium (About this soundlisten (help·info); English translation: On the Revolutions of the Heavenly Spheres) is the seminal work on the heliocentric theory of the astronomer Nicolaus Copernicus (1473–1543) of the Polish Renaissance. The book, first printed in 1543 in Nuremberg, Holy Roman Empire, offered an alternative model of the universe to Ptolemy's geocentric system, which had been widely accepted since ancient times.` (constraints: none; covered=False)
  -> `What is the seminal work on the heliocentric theory of Nicolaus Copernicus, and when and where was it first printed?`
- fact `List A cricket is a classification of the limited-overs (one-day) form of the sport of cricket, with games lasting up to eight hours.` (constraints: as of 1978; covered=False)
  -> `What is the classification of List A cricket and how long do its games typically last?`
- fact `9–1–1 is an American procedural television series created by Ryan Murphy,[3] Brad Falchuk, and Tim Minear for the Fox Broadcasting Company. The series follows the lives of Los Angeles first responders: police officers, paramedics, firefighters, and dispatchers.` (constraints: none; covered=False)
  -> `Who are the creators of the 9–1–1 television series?`
- fact `The race was first organized in 1903 to increase sales for the newspaper L'Auto[2] and is currently run by the Amaury Sport Organisation.` (constraints: none; covered=False)
  -> `When was the race first organized and who currently runs it?`
- fact `The twin boys are named Gregory and Matthew (Ray Romano's actual sons' names) and in the pilot are played by the Ferreira triplets, although only two are seen at any one time.` (constraints: as of 2007; covered=False)
  -> `What are the names of Ray Romano's actual sons who appear as the twin boys in the pilot, and who played them?`
- fact `The Seljuk empire was founded in 1037 by Tughril (990–1063) and his brother Chaghri (989–1060)...The Seljuk empire began to decline in the 1140s, and by 1194 had been supplanted by the Khwarazmian Empire.` (constraints: as of 2001; covered=False)
  -> `When did the Seljuk empire begin to decline?`
- fact `The race was first organized in 1903 to increase sales for the newspaper L'Auto[2] and is currently run by the Amaury Sport Organisation.` (constraints: none; covered=False)
  -> `When was the race first organized and who currently runs it?`
- fact `On 4 September 1988, Galway won the championship following a 1-15 to 0-14 defeat of Tipperary in the All-Ireland final.` (constraints: none; covered=False)
  -> `When did Galway win the championship with a score of 1-15 to 0-14 against Tipperary?`
- fact `Shameless is set in Chicago's South Side[1] and tells the story of an alcoholic father, Frank Gallagher, and his six children who take care of each other and create better lives despite Frank's poor influence. 53	5	"Rite of Passage"	Alex Graves	Etan Frankel	February 8, 2015	1.64` (constraints: as of 1986; covered=False)
  -> `What is the setting of the TV show Shameless?`
- fact `The Book of Genesis (Bereishit in Hebrew) [a], also known as the First Book of Moses,[1] is the first book of the Hebrew Bible and the Christian Old Testament.[2]` (constraints: as of 1988; covered=False)
  -> `What was the date of the last All-Ireland Senior Hurling Championship victory for Galway prior to 2017?`
- fact `The National Commission for Scheduled Castes is an Indian constitutional body under the jurisdiction of Ministry of Social Justice and Empowerment , Government of India established with a view to provide safeguards against the exploitation of Scheduled Castes and Anglo Indian communities to promote and protect their social, educational, economic and cultural interests, special provisions were made in the Constitution.` (constraints: none; covered=False)
  -> `What is the National Commission for Scheduled Castes and what is its constitutional status and purpose in India?`
- fact `They were inducted into the Rock and Roll Hall of Fame as part of the inaugural class of 1986, and into the Country Music Hall of Fame in 2001.` (constraints: as of 1880; covered=False)
  -> `Were they inducted into the Rock and Roll Hall of Fame as part of the inaugural class of 1986, and into the Country Music Hall of Fame in 2001?`
- fact `The car driven by Grace Kelly was a metallic blue 1953 Sunbeam Alpine Mk I.` (constraints: as of 1986; covered=False)
  -> `What was the make, model, and color of the car driven by Grace Kelly as of 1986?`
- fact `Super Troopers is a 2001 American comedy film directed by Jay Chandrasekhar and written by and starring the Broken Lizard comedy group (Chandrasekhar, Kevin Heffernan, Steve Lemme, Paul Soter and Erik Stolhanske).` (constraints: as of 1957; covered=False)
  -> `What is the title and year of the 2001 American comedy film directed by Jay Chandrasekhar and written by the Broken Lizard comedy group?`
- fact `Zooey Deschanel as Lani Aliikai, a Gentoo penguin and Cody's love interest.` (constraints: none; covered=False)
  -> `Who is Zooey Deschanel and what character does she play in the context of a Gentoo penguin and Cody's love interest?`
- fact `The Book of Genesis (Bereishit in Hebrew) [a], also known as the First Book of Moses,[1] is the first book of the Hebrew Bible and the Christian Old Testament.[2]` (constraints: as of 2016; covered=False)
  -> `What is the title of the first episode of Season 1 of This Is Us?`
- fact `They were inducted into the Country Music Hall of Fame in 2001, and Charlie died of cancer in 2011.` (constraints: none; covered=False)
  -> `When were they inducted into the Country Music Hall of Fame and when did Charlie die?`
- fact `The Vietnam Veterans Memorial is a U.S. national memorial in Washington, D.C., honoring service members of the U.S. armed forces who fought in the Vietnam War.` (constraints: none; covered=False)
  -> `What is the location of the Vietnam Veterans Memorial?`
- fact `Independence Day (also known as ID and ID4) is a franchise of American science fiction action films that started with Independence Day in 1996, which was eventually followed by the sequel, Independence Day: Resurgence in 2016.` (constraints: as of 1953; covered=False)
  -> `What is the make and model of the car featured in the 1953 film To Catch a Thief?`
- fact `The twin boys are named Gregory and Matthew (Ray Romano's actual sons' names) and in the pilot are played by the Ferreira triplets, although only two are seen at any one time. In all other episodes of the series, the twins are named Michael and Geoffrey and are played by Sullivan and Sawyer Sweeten.` (constraints: none; covered=False)
  -> `What are the names of the twin boys in the pilot and how are they played?`

## B5 -- full chain (Granite + TRUE co-resident)

Attempted 60 cases; 49 completed the chain, 11 raised (see chain-error notes below). Rates below are over the completed cases.

| diagnostic | value |
|---|---|
| draft claims / faithful (verified) | 71 / 69 |
| supported | 23/69 (0.333) |
| unsupported (dropped by repair) | 46/69 (0.667) |
| contradicted | 0 (expected 0 -- TRUE is binary, no contradiction class) |
| entity mismatches caught | 47 |
| completeness gaps found | 88 |
| gaps patched by recheck | 42/88 (0.477) |
| repair changed the answer | 33/49 (0.673) |
| honest abstention (empty answer) | 28/49 (0.571) |
| GenerationResult contract held | 49/49 (1.000) |
| ms/case | 9586 |

### Chain-error notes (first real run)

- `7150599999316106461`: ValidationError: 1 validation error for EvidenceRecheckResult
  Value error, not-found result cannot contain an answer or evidence [type=value_error, input_value={'required_fact': 'They w...50599999316106461::d4')}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/
- `743600309736467122`: ValueError: recheck output must contain a boolean 'found'
- `6962706727076805207`: ValueError: LLM output must be valid JSON
- `732619765350082410`: ValueError: claim 'claim-2' source_text is not in answer_text
- `4474597907223412132`: ValueError: claim 'claim-2' source_text is not in answer_text
- `-5333049627570569397`: ValueError: recheck output must contain a boolean 'found'
- `7797208395147217894`: ValueError: claim 'claim-2' source_text is not in answer_text
- `-728189424752983312`: ValueError: claim 'claim-2' source_text is not in answer_text
- `7473890225580353748`: ValueError: claim 'claim-2' source_text is not in answer_text
- `-534703981297361693`: ValueError: claim 'claim-2' source_text is not in answer_text
- `-1892104729672825209`: ValueError: claim 'claim-2' source_text is not in answer_text

