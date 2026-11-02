# VoiceTutor

A voice-first study assistant for visually impaired and dyslexic students: the student uploads their notes, asks by voice, and hears grounded answers with page citations.

## Language

**Notes**:
The document a student uploads in a session (PDF, DOCX or TXT); the only source a **Grounded answer** may draw from.
_Avoid_: corpus, knowledge base

**Session**:
One browser tab's lifetime of use; owns its **Notes** index and conversation, never shared or stored.
_Avoid_: user, account

**Grounded answer**:
A reply built only from retrieved passages of the **Notes**, citing page numbers.

**Extractive answer**:
A **Grounded answer** made by reading retrieved passages aloud verbatim, used when no LLM is available.
_Avoid_: fallback answer

**Section**:
A stretch of the **Notes** found by searching for the name the student says, then widened to the neighbouring passages on the same topic; not tied to headings in the file.
_Avoid_: chapter, heading

**General-knowledge answer**:
A reply not drawn from the **Notes**, allowed only when the student opts in, and always labelled "this is not from your notes".

## Relationships

- A **Session** has at most one **Notes** at a time
- A reply is a **Grounded answer** (generated or **Extractive**), a **General-knowledge answer**, or a "not found in your notes" refusal

## Flagged ambiguities

- None yet.
