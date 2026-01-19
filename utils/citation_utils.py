"""
Citation Utilities for UTI Prediction Pipeline
===============================================
Functions for tracking and verifying citations.
CRITICAL: No citation should be used without verification!
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import hashlib


class CitationTracker:
    """
    Track and verify citations for the manuscript.

    Ensures zero-tolerance for unsourced citations.
    """

    def __init__(self, project_root: Path):
        """
        Initialize citation tracker.

        Parameters
        ----------
        project_root : Path
            Root directory of the project
        """
        self.project_root = Path(project_root)
        self.references_dir = self.project_root / 'References'
        self.papers_dir = self.references_dir / 'papers'
        self.log_file = self.references_dir / 'citations_log.md'
        self.bib_file = self.references_dir / 'bibliography.bib'

        # Ensure directories exist
        self.papers_dir.mkdir(parents=True, exist_ok=True)

        # Citation categories
        self.categories = [
            'epidemiology',
            'machine_learning',
            'clinical',
            'ode_parameters',
            'methodology'
        ]

        for cat in self.categories:
            (self.papers_dir / cat).mkdir(exist_ok=True)

        # Load existing citations
        self.citations = self._load_citations()

    def _load_citations(self) -> Dict:
        """Load existing citations from log file."""
        citations = {}

        if self.log_file.exists():
            # Parse existing log
            pass  # Would parse markdown log

        return citations

    def add_citation(self,
                    key: str,
                    title: str,
                    authors: str,
                    year: int,
                    journal: str,
                    category: str,
                    pdf_filename: Optional[str] = None,
                    doi: Optional[str] = None,
                    url: Optional[str] = None,
                    notes: str = '',
                    verified: bool = False) -> Dict:
        """
        Add a new citation to the tracker.

        Parameters
        ----------
        key : str
            BibTeX key (e.g., 'smith2023uti')
        title : str
            Paper title
        authors : str
            Author list
        year : int
            Publication year
        journal : str
            Journal name
        category : str
            Paper category
        pdf_filename : str, optional
            PDF filename in papers directory
        doi : str, optional
            DOI
        url : str, optional
            URL
        notes : str
            Usage notes
        verified : bool
            Whether PDF has been verified

        Returns
        -------
        dict
            Citation record
        """
        if category not in self.categories:
            raise ValueError(f"Invalid category. Use one of: {self.categories}")

        citation = {
            'key': key,
            'title': title,
            'authors': authors,
            'year': year,
            'journal': journal,
            'category': category,
            'pdf_filename': pdf_filename,
            'doi': doi,
            'url': url,
            'notes': notes,
            'verified': verified,
            'added_date': datetime.now().isoformat(),
            'pdf_exists': False
        }

        # Check if PDF exists
        if pdf_filename:
            pdf_path = self.papers_dir / category / pdf_filename
            citation['pdf_exists'] = pdf_path.exists()

        self.citations[key] = citation

        return citation

    def verify_citation(self, key: str) -> bool:
        """
        Verify that a citation has a valid PDF.

        Parameters
        ----------
        key : str
            Citation key

        Returns
        -------
        bool
            Whether citation is verified
        """
        if key not in self.citations:
            return False

        citation = self.citations[key]

        if not citation.get('pdf_filename'):
            return False

        pdf_path = self.papers_dir / citation['category'] / citation['pdf_filename']

        if pdf_path.exists():
            self.citations[key]['verified'] = True
            self.citations[key]['pdf_exists'] = True
            return True

        return False

    def get_unverified_citations(self) -> List[Dict]:
        """Get list of unverified citations."""
        return [c for c in self.citations.values() if not c.get('verified')]

    def get_citation_needed_placeholders(self,
                                        text: str) -> List[str]:
        """
        Find all [CITATION NEEDED] placeholders in text.

        Parameters
        ----------
        text : str
            Text to search

        Returns
        -------
        list
            List of topics needing citations
        """
        import re
        pattern = r'\[CITATION NEEDED:?\s*([^\]]*)\]'
        matches = re.findall(pattern, text)
        return matches

    def generate_bibtex(self, key: str) -> str:
        """
        Generate BibTeX entry for a citation.

        Parameters
        ----------
        key : str
            Citation key

        Returns
        -------
        str
            BibTeX entry
        """
        if key not in self.citations:
            raise KeyError(f"Citation '{key}' not found")

        c = self.citations[key]

        bibtex = f"""@article{{{c['key']},
  author = {{{c['authors']}}},
  title = {{{c['title']}}},
  journal = {{{c['journal']}}},
  year = {{{c['year']}}},"""

        if c.get('doi'):
            bibtex += f"\n  doi = {{{c['doi']}}},"
        if c.get('url'):
            bibtex += f"\n  url = {{{c['url']}}},"

        bibtex += "\n}"

        return bibtex

    def export_bibliography(self) -> str:
        """Export all citations as BibTeX file."""
        entries = []
        for key in sorted(self.citations.keys()):
            entries.append(self.generate_bibtex(key))

        bibtex_content = '\n\n'.join(entries)

        with open(self.bib_file, 'w', encoding='utf-8') as f:
            f.write(bibtex_content)

        return bibtex_content

    def generate_log(self) -> str:
        """Generate markdown citation log."""
        lines = [
            '# Citation Verification Log',
            f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
            '',
            '## Summary',
            f'- Total citations: {len(self.citations)}',
            f'- Verified: {sum(1 for c in self.citations.values() if c.get("verified"))}',
            f'- Pending verification: {sum(1 for c in self.citations.values() if not c.get("verified"))}',
            '',
            '## Citations by Category',
            ''
        ]

        for category in self.categories:
            cat_citations = [c for c in self.citations.values()
                          if c.get('category') == category]

            if cat_citations:
                lines.append(f'### {category.replace("_", " ").title()}')
                lines.append('')

                for c in cat_citations:
                    status = '[x]' if c.get('verified') else '[ ]'
                    lines.append(f"- {status} **{c['key']}**: {c['title']} ({c['year']})")
                    if c.get('notes'):
                        lines.append(f"  - Notes: {c['notes']}")
                    lines.append('')

        log_content = '\n'.join(lines)

        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(log_content)

        return log_content

    def audit(self) -> Dict:
        """
        Perform citation audit.

        Returns
        -------
        dict
            Audit results
        """
        results = {
            'total': len(self.citations),
            'verified': 0,
            'unverified': [],
            'missing_pdfs': [],
            'by_category': {}
        }

        for key, citation in self.citations.items():
            if citation.get('verified'):
                results['verified'] += 1
            else:
                results['unverified'].append(key)

            if citation.get('pdf_filename') and not citation.get('pdf_exists'):
                results['missing_pdfs'].append(key)

            cat = citation.get('category', 'uncategorized')
            if cat not in results['by_category']:
                results['by_category'][cat] = {'total': 0, 'verified': 0}
            results['by_category'][cat]['total'] += 1
            if citation.get('verified'):
                results['by_category'][cat]['verified'] += 1

        return results


def create_citation_placeholder(topic: str) -> str:
    """
    Create a citation needed placeholder.

    Use this instead of making up citations!

    Parameters
    ----------
    topic : str
        Topic needing citation

    Returns
    -------
    str
        Placeholder text
    """
    return f'[CITATION NEEDED: {topic}]'


def search_citation_databases(query: str,
                             databases: List[str] = None) -> str:
    """
    Generate search URLs for citation databases.

    Parameters
    ----------
    query : str
        Search query
    databases : list, optional
        Which databases to search

    Returns
    -------
    str
        Markdown with search links
    """
    if databases is None:
        databases = ['pubmed', 'google_scholar', 'semantic_scholar']

    query_encoded = query.replace(' ', '+')

    urls = {
        'pubmed': f'https://pubmed.ncbi.nlm.nih.gov/?term={query_encoded}',
        'google_scholar': f'https://scholar.google.com/scholar?q={query_encoded}',
        'semantic_scholar': f'https://www.semanticscholar.org/search?q={query_encoded}'
    }

    lines = [f'## Search for: "{query}"', '']
    for db in databases:
        if db in urls:
            lines.append(f'- [{db.replace("_", " ").title()}]({urls[db]})')

    return '\n'.join(lines)
