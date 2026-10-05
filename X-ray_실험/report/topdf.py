from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(); pg.goto('file:///home/claude/kamp/X-ray_실험/report/report.html'); pg.wait_for_timeout(800)
    pg.pdf(path='/home/claude/kamp/X-ray_실험/report/report.pdf', format='A4', print_background=True, display_header_footer=True,
           header_template='<div></div>', footer_template='<div style="font-size:8pt;width:100%;text-align:center;color:#666"><span class="pageNumber"></span></div>',
           margin={'top':'20mm','bottom':'20mm','left':'18mm','right':'18mm'})
    b.close()
