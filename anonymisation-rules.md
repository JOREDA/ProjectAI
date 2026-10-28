# Anonymisation Rules & Guidelines

This document specifies the privacy redaction and generalisation rules for employee feedback at Navrang Systems. The pipeline applies two sequential layers of privacy protection.

---

## Privacy Layer 1: Direct Identifiers (Call A)

Direct identifiers point specifically to an individual or small traceable group and must be removed or replaced with neutral placeholders without altering the underlying complaint, meaning, sentiment, or severity.

### Direct Identifiers to Remove or Replace
1. **Person names (Manager or Colleague)**: Replace with neutral roles such as `"my manager"`, `"a colleague"`, `"my team lead"`, or `"senior leadership"`.
2. **Writer's own name or self-identification**: Replace with `"I"` or omit self-naming prefixes.
3. **Employee IDs**: Any string matching `NV\d{5}` (e.g., `NV01482`) must be replaced with `"[ID removed]"` or omitted.
4. **Project codes**: Project identifiers (such as `PRJ-ATLAS-2291`, `PRJ-KESTREL-118`, `ORBIS-INS-07`, `HELIX-RTL-342`, `PRJ-NOVA-55`, `SAKSHAM-OPS-12`, `PRJ-MERIDIAN-9`, `VAAYU-CX-204`) must be replaced with `"our project"`, `"the project"`, or `"a client engagement"`.
5. **Client names**: External company names (such as `Orbis Insurance`, `Helix Retail`, `Kestrel Bank`, `Meridian Health`, `Vaayu Airlines`) must be replaced with `"the client"` or `"our client"`.
6. **Office locations and Cities**: Pune, Bengaluru, Mumbai, Chennai, Hyderabad, Delhi NCR, Gurgaon, Noida must be generalised to `"our office"` or omitted if context permits.
7. **Building names & Floors**: Details like `"4th floor"`, `"Tower B 7th floor"`, `"Hinjewadi Phase 2"`, `"Whitefield annex"`, `"Guindy campus"`, `"Cyber City Block C"` must be replaced with `"the office"` or `"our facility"`.
8. **Specific dates & exact timestamps**: Exact calendar dates (e.g., `"14th October 2026"`, `"last Tuesday at 3:15 PM"`) must be generalised to relative temporal markers like `"recently"`, `"last month"`, or `"during a recent sprint"`.

### What Must Be Preserved in Layer 1
- The factual substance of the feedback/complaint.
- The tone and emotional intensity (frustration, concern, dissatisfaction).
- The specific practice or policy being criticised (e.g., overtime, review bias, unfair targets).
- Target distinction: whether it is a systemic policy issue, interpersonal conduct, or both.

---

## Privacy Layer 2: Indirect Clues & Re-identification Review (Call B)

Even without names, colleagues or managers reading feedback may identify the writer from unique circumstances, isolated roles, specific incidents, or signature writing styles. Layer 2 reads the text from the perspective of a team member and asks: *"Could someone guess who wrote this?"*

### Indirect Clues to Generalise
1. **Unique roles or demographics**:
   - *"I am the only female developer on the night shift in Pune"* → Generalise to: *"I work night shifts on our technical team"*.
   - *"As the junior-most member who joined from IIT Bombay in August"* → Generalise to: *"As a recent joiner"*.
2. **Singular memorable incidents**:
   - *"I'm the one who raised the payroll error in the town hall last month"* → Remove the identifying incident and retain the core grievance: *"I have experienced recurring issues with promotion cycles"*.
   - *"After my motorcycle accident last Friday when my laptop broke"* → Generalise to: *"After an unexpected emergency when my equipment broke"*.
3. **Idiosyncratic phrasing & catchphrases**:
   - Distinctive personal idioms, unique greetings, or repeated signature mottoes must be normalised into standard workplace English.
4. **Hinglish and Colloquialisms**:
   - Preserve natural sentiment while smoothing distinct individual colloquial habits:
   - *"Manager roz 10 baje ping karta hai, bole toh no work-life balance"* → *"My manager messages late every night around 10 PM, leaving no work-life balance."*

### Re-identification Risk Assessment
- **Low**: Generic team experiences shared by many peers; no unique clues remain.
- **Medium**: Mentions a relatively small group, uncommon workflow, or recent department event that slightly narrows the writer's identity.
- **High**: Contains a singular role, unique public altercation, or isolated event that clearly points to one person even after direct names are stripped.

---

## Worked Examples

### Example 1: Routine Workload
- **Raw Input**: "My manager keeps scheduling calls after 9 pm for the release of our project and expects replies on WhatsApp on Sundays."
- **Layer 1 Output**: "My manager keeps scheduling calls after 9 pm for the release of our project and expects replies on WhatsApp on Sundays."
- **Layer 2 Output**: "My manager keeps scheduling calls after 9 pm for the release of our project and expects replies on WhatsApp on Sundays."
- **PII Found**: `[]`
- **Re-ID Risk**: `Low`
- **Re-ID Reason**: `""`

### Example 2: Names and Project Codes
- **Raw Input**: "Pooja told me on PRJ-ATLAS-2291 that Vikram will handle the client demo for Orbis Insurance in Bengaluru."
- **Layer 1 Output**: "My manager told me on our project that a colleague will handle the client demo for the client in our office."
- **Layer 2 Output**: "My manager informed me that another colleague will handle the upcoming client demo instead of me."
- **PII Found**: `["Person name (manager)", "Person name (colleague)", "Project code", "Client name", "Office location"]`
- **Re-ID Risk**: `Low`
- **Re-ID Reason**: `""`

### Example 3: Re-identifiable Situation (Edge Case)
- **Raw Input**: "I'm the one who raised the payroll error in the town hall last month. I've been 'next cycle' for promotion three cycles in a row now."
- **Layer 1 Output**: "I'm the one who raised the payroll error in the town hall last month. I've been 'next cycle' for promotion three cycles in a row now."
- **Layer 2 Output**: "I have been told I am in line for promotion in the 'next cycle' for three consecutive cycles now without any progress."
- **PII Found**: `[]`
- **Re-ID Risk**: `High`
- **Re-ID Reason**: `"Unique role/situation identifies the writer even with names removed"`

### Example 4: Hinglish & Informal Style
- **Raw Input**: "Rahul sir team meeting me chillate rehte hain sabke samne in Hinjewadi Phase 2. Bahut insult feel hota hai."
- **Layer 1 Output**: "My manager keeps shouting in team meetings in front of everyone in our facility. It feels very insulting."
- **Layer 2 Output**: "My manager frequently shouts at team members in front of everyone during meetings, which is deeply humiliating."
- **PII Found**: `["Person name (manager)", "Office floor/building"]`
- **Re-ID Risk**: `Low`
- **Re-ID Reason**: `""`
