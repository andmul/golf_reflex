## 2024-03-24 - Reflex Select & Checkbox Accessibility
**Learning:** Reflex `rx.select` and `rx.checkbox` components often lack programmatic association with their visual labels, especially when `rx.text` is used separately.
**Action:** Always ensure interactive elements like `rx.select` and `rx.checkbox` have an `aria_label` attribute or are properly associated with a label using `html_for` (or similar) or wrapped in a labeled control.
