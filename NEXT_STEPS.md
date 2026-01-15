# Next Steps - Quick Action Guide

**Project:** ML-Driven Functional Genomics System  
**Updated:** January 7, 2026

---

## ✅ COMPLETED

- [x] Project idea selected and validated
- [x] Detailed proposal written
- [x] Abstract completed (ready for advisor outreach)
- [x] System architecture designed
- [x] Advisor targeting strategy developed
- [x] Documentation structure established

---

## 🎯 IMMEDIATE PRIORITIES (This Week)

### 1. Find & Contact Faculty Advisors **[CRITICAL - Due before first class]**

**Action Items:**
1. Review Dr. Marte Zorrilla's recommended faculty list (check email/Canvas)
2. Search UF departments for potential advisors:
   - Biostatistics: https://biostat.ufl.edu/faculty/
   - CISE: https://www.cise.ufl.edu/people/faculty/
   - Genetics Institute faculty
   - Molecular Genetics & Microbiology
   - Biomedical Engineering

3. Create your target list (use `advisor-targeting-strategy.md` worksheet):
   - Identify 5-7 potential advisors
   - Research their recent publications
   - Score them using the match criteria

4. Prepare personalized emails:
   - Use templates in `advisor-targeting-strategy.md`
   - Customize for each faculty member
   - Attach your abstract (from `preliminary-idea.md`)
   - Emphasize: "ML-heavy capstone with deployable system"

5. Send emails to top 3-5 advisors
   - Best timing: Tuesday-Thursday morning
   - Follow up in 3-4 days if no response

**Files to Use:**
- `project-proposal/preliminary-idea.md` - Your abstract
- `project-proposal/advisor-targeting-strategy.md` - Templates & strategy
- `project-proposal/advisor-email-template.md` - Contact tracking

**Target:** Secure at least one advisor conversation before first class

---

### 2. Prepare for First Class Presentation

**What You Need:**
- Elevator pitch (30 seconds): Problem → Approach → Impact
- Key points to cover:
  - Problem: Non-coding variant interpretation challenge
  - Solution: End-to-end ML system with deep learning
  - Data: ENCODE, GTEx (all public, ethical)
  - Deliverables: Pipeline + Models + Interface + Documentation
  - Timeline: Phased approach over semester

**Practice:**
- Be able to explain to technical AND non-technical audiences
- Anticipate questions about scope, feasibility, data access

---

### 3. Begin Literature Review

**Priority Papers to Read:**

1. **DeepSEA (Nature Methods 2015)** - Foundation for CNN approach
   - Zhou & Troyanskaya
   - Understand their architecture and evaluation

2. **Basenji (Genome Research 2018)** - Advanced sequence modeling
   - Kelley et al.
   - Cross-species and cell-type generalization

3. **ENCODE Overview (Nature 2012)** - Data source understanding
   - ENCODE Consortium
   - Types of functional data available

4. **DNABERT (Bioinformatics 2021)** - Pretrained models
   - Ji et al.
   - Transformer approach to DNA sequences

**Where to Start:**
- Create summaries in `research/` folder
- Use paper template from `research/README.md`
- Focus on: methodology, evaluation metrics, datasets used

---

## 📅 WEEK 2 PRIORITIES

### Technical Setup
- [ ] Set up GitHub repository (private initially)
- [ ] Configure Python environment (conda/venv)
- [ ] Install base dependencies
- [ ] Test GPU access (Colab Pro or HiPerGator)

### Data Exploration
- [ ] Access ENCODE portal, browse chromatin data
- [ ] Download small benchmark dataset (DeepSEA)
- [ ] Familiarize with file formats (BED, BigWig, VCF)
- [ ] Create data exploration notebook

### Advisor Relationship
- [ ] Schedule first advisor meeting
- [ ] Prepare questions about dataset selection
- [ ] Discuss scope adjustments if needed
- [ ] Clarify communication frequency

---

## 📊 TRACKING YOUR PROGRESS

**Use these files regularly:**

1. **Daily/Weekly:** `milestones/progress-tracker.md`
   - Update weekly log
   - Track accomplishments and blockers
   - Document advisor meetings

2. **As Needed:** `project-proposal/advisor-email-template.md`
   - Track faculty contact attempts
   - Record responses and next steps

3. **Continuous:** `research/` folder
   - Add paper summaries as you read
   - Build your literature review

---

## 🚨 POTENTIAL BLOCKERS & SOLUTIONS

| Blocker | Solution |
|---------|----------|
| No advisor by first class | Have 2-3 conversations in progress; Dr. Marte Zorrilla can help |
| Faculty says project too ambitious | Reference system architecture - show it's well-scoped |
| Concerned about genomics knowledge | Emphasize you'll learn domain; they guide interpretation |
| GPU access unclear | Multiple options: Colab Pro ($10/mo), HiPerGator (free), AWS credits |
| Dataset seems overwhelming | Start with DeepSEA benchmark - already preprocessed |

---

## 💡 TIPS FOR SUCCESS

### When Talking to Advisors:
✅ **DO:**
- Emphasize your ML expertise
- Show you've done homework (read their papers)
- Ask for guidance, not permission
- Be flexible but confident

❌ **DON'T:**
- Ask them to teach you ML
- Be vague about deliverables
- Oversell capabilities
- Underestimate timeline

### Project Management:
- **Think in milestones:** Plan → Baseline → Deep Learning → Evaluation → Deploy
- **Document everything:** Future-you will thank present-you
- **Fail fast:** Test ideas quickly, pivot if needed
- **Communicate often:** Weekly advisor updates minimum

---

## 📞 WHO TO CONTACT FOR WHAT

**Dr. Edwin Marte Zorrilla (Course Instructor):**
- Course requirements and expectations
- Advisor recommendation assistance
- Milestone deadlines and deliverables
- General course questions

**Faculty Advisor (Once secured):**
- Domain-specific questions
- Dataset and approach validation
- Biological interpretation
- Technical guidance

**HiPerGator Support (if using):**
- GPU allocation requests
- Environment setup
- Job submission help

**Peers in Class:**
- Brainstorming sessions
- Feedback on presentations
- Study groups for shared challenges

---

## 🎯 GOALS FOR END OF JANUARY

By end of January, you should have:
- ✅ Secured faculty advisor
- ✅ Completed initial literature review (5-10 key papers)
- ✅ Set up development environment
- ✅ Downloaded and explored benchmark dataset
- ✅ Created initial project plan with advisor approval
- ✅ Built baseline model (simple ML)
- ✅ Established regular advisor meeting schedule

---

## 📁 KEY FILES QUICK REFERENCE

```
Your Workspace/
│
├── GETTING_STARTED.md              ← Overview guide
├── THIS FILE (next-steps.md)       ← Action items
│
├── project-proposal/
│   ├── preliminary-idea.md         ← Your complete proposal & abstract
│   ├── brainstorming.md            ← Project idea documentation
│   ├── advisor-targeting-strategy.md ← How to find advisors
│   └── advisor-email-template.md   ← Email templates & tracking
│
├── milestones/
│   └── progress-tracker.md         ← Weekly progress log
│
├── documentation/
│   └── system-architecture.md      ← Technical design document
│
└── research/
    └── README.md                   ← Literature review guide
```

---

## 🔄 WEEKLY ROUTINE (Once Project Starts)

**Monday:**
- Review last week's progress
- Set this week's goals
- Update progress tracker

**During Week:**
- Work on current milestone tasks
- Document experiments/decisions
- Read 1-2 papers

**Friday:**
- Weekly advisor meeting (or async update)
- Review accomplishments
- Identify blockers
- Plan next week

**Continuous:**
- Commit code changes with clear messages
- Update documentation as you code
- Track experiments in MLflow/W&B

---

## ✨ REMEMBER

This is a **capstone project** - it should showcase your skills:
- ML expertise ✓
- System design thinking ✓
- Scientific rigor ✓
- Communication ability ✓
- Project management ✓

**You've got this!** The planning is done, now execute step by step. Focus on this week's priorities first.

---

**Questions?** Review your documentation files or contact Dr. Marte Zorrilla.

**Ready?** Start with action item #1: Find your advisor! 🚀
