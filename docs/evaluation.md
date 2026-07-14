# Evaluation agreement

## Runtime Query and evaluation labels

The Pipeline receives only a Query: query ID plus question text. Gold answers and gold
evidence IDs stay outside the Pipeline and are used only by evaluation. This prevents the
system from seeing the answer or evidence labels while it is being tested.

## Module development

Each module team may choose datasets appropriate to its own objective.
The chosen training, development, and test roles must be written in that module's plan.

- Retriever data normally provides a corpus, questions, and relevant document labels.
- Selector data normally provides questions, candidate pools, and selected-evidence labels.
- Generator data normally provides questions, evidence, and reference answers.

Module-specific datasets do not need to be identical.

## Complete-Pipeline development

The team uses agreed development data to connect Retriever, Selector, and Generator.
These results are for development, not the final comparison.

A shared end-to-end benchmark should provide a searchable corpus, questions, reference
answers, and gold document or evidence IDs. The real Pipeline must build CandidateSet through
Retriever; a hand-constructed pool that guarantees gold evidence is present cannot prove
end-to-end performance.

After the clean baseline is accepted, dataset selection and a dataset adapter are the next
implementation phase. Different source schemas will be translated into a common evaluation
record while only the Query is passed into the Pipeline.

## FinanceBench

The team agrees that FinanceBench is reserved for final comparison of the complete RAG
system against other RAG systems.

It will not be used for Retriever, Selector, or Generator training, development, tuning,
or process testing. This is a team workflow agreement; the code does not implement a
FinanceBench access restriction.

The other two possible final benchmarks have not been selected.
