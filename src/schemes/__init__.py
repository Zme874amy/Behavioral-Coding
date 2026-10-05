"""Coding schemes and the mappings between them: the single source of truth.

Every code list, tier grouping, label alias and cross-scheme mapping used by the
project lives here. Other modules import from this package; none defines its own
copy (docs/RERUN_PLAN.md P1.3).

    schemes.misc      MISC 2.5 (+ AutoMISC's AC+/- and T1 groups): our training scheme
    schemes.miti      MITI 4.2.1 behaviour codes
    schemes.annomi    AnnoMI utterance attributes
    schemes.mappings  Welivita -> MITI, MISC <-> MITI, MISC -> AnnoMI, Welivita -> MISC

Sources (transcribed from the manuals, not derived from data):
  MISC 2.5   Houck, Moyers, Miller, Glynn & Hallgren (2010): counsellor categories p.16;
             No Code p.14; client categories pp.38-41; MICO/MIIN pp.47-48; globals p.1
  MITI 4.2.1 Moyers, Manuel & Ernst (2015), casaa.unm.edu/assets/docs/miti4_21.pdf
  Welivita   Welivita & Pu (COLING 2022) Table 1 (15 labels adapted from MITI 2.0 / 4.2.1)
  AnnoMI     Wu et al. (Future Internet 2023) Sec. 4 utterance attributes

Mappings return None for a code with no counterpart. They never fall back to a
default code silently; a caller that wants one must say so.
"""
from schemes import annomi, mappings, misc, miti

HANDBOOK = {
    "MISC 2.5": misc.HANDBOOK,
    "MITI 4.2.1": miti.HANDBOOK,
    "Welivita (MITI-derived)": mappings.WELIVITA_HANDBOOK,
    "AnnoMI": annomi.HANDBOOK,
}

__all__ = ["misc", "miti", "annomi", "mappings", "HANDBOOK"]
