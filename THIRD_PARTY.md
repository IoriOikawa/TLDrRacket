# Third-party frontend assets

| Asset | Version | Upstream | License |
| --- | --- | --- | --- |
| `static/vendor/showdown-2.1.0.min.js` | 2.1.0 | [Showdown](https://github.com/showdownjs/showdown/tree/2.1.0) | [MIT](static/vendor/showdown-LICENSE) |
| `static/vendor/purify-3.4.17.min.js` | 3.4.17 | [DOMPurify](https://github.com/cure53/DOMPurify/tree/3.4.17) | [MPL-2.0 OR Apache-2.0](static/vendor/DOMPurify-LICENSE) |

Files are copied from the tagged upstream `dist/` directories without modifying their executable contents; line endings are normalized to LF. To update, review the upstream release, replace the versioned file and license, update `static/index.html`, and rerun the browser checks. No npm install or build step is needed at runtime.

The existing course-style CSS, icon fonts and illustration are retained from the original repository. PDF/text readings are source material, separate from the application's MIT license.
