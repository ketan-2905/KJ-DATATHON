UI Design Specification: Equilibrium.ai (2D Framework)
1. Core Visual Language
Palette: * Background: White (#FFFFFF) with section offsets in Slate-50 (#f8fafc).

Typography: Slate-900 for headings, Slate-500 for body, Indigo-600 for accents.

Borders: Slate-200 (1px solid).

Corner Radius: Extremely rounded. Use rounded-[2.5rem] for main cards and rounded-full for buttons/tags.

Shadows: Very subtle. Use shadow-sm for cards and shadow-2xl shadow-slate-400/20 for floating primary buttons.

2. Typography & Hierarchy
Headers: font-bold tracking-tighter. For the hero, use text-7xl or larger with a leading-[0.85] to create a "tight" editorial feel.

The "Meta" Label: Small, all-caps, bold text used for categories.

Class: text-[10px] font-black uppercase tracking-[0.2em]

The "Italic Hook": Use italicized text within headers or for values to provide a "mathematical/academic" feel.

3. Component Architecture
The Bento Card: * A white container with a Slate-200 border.

Contains a small icon in an Indigo-50 square with rounded-2xl.

Footer of the card often features a 3-column "Metric Row" with vertical dividers (border-x).

Impact Metrics: * A row with a colored indicator dot (w-1.5 h-1.5), a bold label, and a "status" value in all-caps on the right.

Navigation: Sticky bg-white/80 with a heavy backdrop-blur-md and a thin border-b.

4. Interaction Patterns
Hover States: Cards should transition their border color (e.g., hover:border-indigo-200).

Active States: Buttons should utilize active:scale-95 for a tactile, physical feel.

Empty Space: Prioritize whitespace. No element should feel crowded. Use max-w-7xl containers for all content.

5. Iconography
Style: Thin-stroke, geometric icons (Lucide-react preferred).

Usage: Icons are never alone; they are always contained in a styled background or paired with high-tracking labels.