"""Optional browser smoke checks. Start webserver.py, then run this with Playwright."""
import asyncio

from playwright.async_api import async_playwright, expect

URL = 'http://127.0.0.1:5000'


async def main():
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch()
        page = await browser.new_page(viewport={'width': 1000, 'height': 850})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        await page.goto(URL)
        await expect(page.locator('#request-status')).to_be_hidden()
        # Hold a response so the loading state can be observed, including elapsed time.
        gate = asyncio.Event()
        calls = []
        async def delayed(route):
            calls.append(route.request.post_data_json)
            await gate.wait()
            await route.fulfill(json={'response': 'Closures capture their lexical environment. [1]', 'status': 'ok',
                                      'sources': [{'file': 'PLNotes10 - Closures and Environments.pdf', 'page': 2,
                                                   'preview': 'A closure carries its defining environment.',
                                                   'url': '/documents/PLNotes10%20-%20Closures%20and%20Environments.pdf#page=2'}]})
        await page.route('**/api/ask', delayed)
        await page.locator('#chat-input').fill('How do closures work?')
        await page.locator('#chat-input').press('Enter')
        await expect(page.locator('#request-status')).to_be_visible()
        await expect(page.locator('#send-button')).to_be_disabled()
        await page.locator('#chat-form').evaluate("form => form.dispatchEvent(new Event('submit', {cancelable:true}))")
        await expect(page.locator('#request-elapsed')).not_to_have_text('0s')
        assert len(calls) == 1, calls
        gate.set()
        await expect(page.locator('.reading-sources li')).to_have_count(1)
        await expect(page.locator('#request-status')).to_be_hidden()
        await expect(page.locator('#send-button')).to_be_enabled()
        await expect(page.locator('.reading-sources a')).to_have_attribute('href', URL + '/documents/PLNotes10%20-%20Closures%20and%20Environments.pdf#page=2')
        await page.unroute('**/api/ask')

        async def answer(payload, status=200):
            await page.route('**/api/ask', lambda route: route.fulfill(json=payload, status=status))
            await page.locator('#chat-input').fill('A question')
            await page.locator('#send-button').click()
            await expect(page.locator('#request-status')).to_be_hidden()
            await expect(page.locator('#send-button')).to_be_enabled()
            await page.unroute('**/api/ask')

        await answer({'response': 'No relevant material found.', 'sources': [], 'status': 'no_match'})
        await expect(page.locator('.message').last.locator('.reading-sources')).to_have_count(0)
        await expect(page.locator('.message').last).to_contain_text('No relevant reading locations found.')
        await answer({'error': 'The assistant is busy.'}, status=429)
        await expect(page.locator('.message').last.locator('h4')).to_have_text('eLi')
        await expect(page.locator('.message').last).to_contain_text('busy')
        sources = [{'file': f'Notes {i}.pdf', 'page': i+1, 'preview': 'Relevant excerpt', 'url': f'/documents/notes{i}.pdf#page={i+1}'} for i in range(5)]
        await answer({'response': '**Grounded answer**', 'sources': sources, 'status': 'ok'})
        await expect(page.locator('.message').last.locator('.reading-sources li')).to_have_count(5)
        markdown = '# Closures\n\n- Keep the defining environment\n- Can be returned\n\n```racket\n(lambda (x) (+ x 1))\n```\n\n| Concept | Meaning |\n| --- | --- |\n| Closure | Function and environment |'
        markdown_source = {'file': '5400Notes.txt', 'page': None, 'preview': 'Closures',
                           'content': markdown, 'format': 'markdown', 'url': '/documents/5400Notes.txt'}
        await answer({'response': 'A grounded answer [1]', 'sources': [markdown_source], 'status': 'ok'})
        excerpt = page.locator('.message').last.locator('.source-excerpt')
        await expect(excerpt).to_have_count(1)
        await expect(excerpt).to_have_attribute('open', '')
        await expect(page.locator('.message').last.locator('.source-preview')).to_have_count(0)
        await expect(excerpt.locator('h1')).to_have_text('Closures')
        await expect(excerpt.locator('ul li')).to_have_count(2)
        await expect(excerpt.locator('pre code')).to_contain_text('(lambda (x) (+ x 1))')
        await expect(excerpt.locator('table')).to_have_count(1)
        poisoned_source = dict(markdown_source, content='<img src=x onerror="window.sourcePwned=1"><script>window.sourcePwned=1</script>[bad](javascript:alert(1))')
        await answer({'response': 'Excerpt with unsafe HTML', 'sources': [poisoned_source], 'status': 'ok'})
        await expect(page.locator('.message').last.locator('.source-content img, .source-content script, .source-content a[href^="javascript:"]')).to_have_count(0)
        assert await page.evaluate('window.sourcePwned') is None
        plain_source = dict(markdown_source, file='Notes.pdf', format='text', content='# Literal PDF text\n<b>not HTML</b>')
        await answer({'response': 'PDF excerpt', 'sources': [plain_source], 'status': 'ok'})
        await expect(page.locator('.message').last.locator('.source-content')).to_have_text(plain_source['content'])
        await expect(page.locator('.message').last.locator('.source-content h1, .source-content b')).to_have_count(0)
        await page.evaluate('window.savedPurify = window.DOMPurify; window.DOMPurify = undefined')
        await answer({'response': '**Fallback answer**', 'sources': [markdown_source], 'status': 'ok'})
        await expect(page.locator('.message').last.locator('.source-content')).to_have_text(markdown)
        await page.evaluate('window.DOMPurify = window.savedPurify; delete window.savedPurify')
        await answer({'response': '<img src=x onerror="window.pwned=1"><script>window.pwned=1</script>[bad](javascript:alert(1))', 'sources': [], 'status': 'ok'})
        assert await page.evaluate('window.pwned') is None
        await expect(page.locator('.message').last.locator('img, script, a[href^="javascript:"]')).to_have_count(0)
        await page.locator('#chat-input').fill('<img src=x onerror="window.pwned=1">')
        await page.route('**/api/ask', lambda route: route.fulfill(json={'response': 'Safe', 'sources': [], 'status': 'ok'}))
        await page.locator('#send-button').click()
        await expect(page.locator('#send-button')).to_be_enabled()
        assert await page.evaluate('window.pwned') is None
        await page.unroute('**/api/ask')
        # Browser virtual time makes the real 90-second abort path quick to test.
        await page.clock.install()
        await page.route('**/api/ask', lambda route: asyncio.sleep(0))
        await page.locator('#chat-input').fill('Timeout test')
        await page.locator('#send-button').click()
        await expect(page.locator('#request-status')).to_be_visible()
        await page.clock.fast_forward(91000)
        await expect(page.locator('#send-button')).to_be_enabled()
        await expect(page.locator('.message').last).to_contain_text('took too long')
        await page.unroute('**/api/ask')
        await page.set_viewport_size({'width': 390, 'height': 844})
        assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        await page.emulate_media(reduced_motion='reduce')
        assert await page.locator('.spinner').evaluate("element => getComputedStyle(element).animationName") == 'none'
        assert not errors, errors
        print('Browser checks passed: waiting, elapsed time, duplicate guard, 0/1/5 sources, errors, timeout, Markdown excerpts, plain PDF excerpts, sanitizer fallback, XSS, mobile, reduced motion.')
        await browser.close()


if __name__ == '__main__':
    asyncio.run(main())
