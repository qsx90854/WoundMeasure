from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from pathlib import Path
out=Path(r'D:\Lightglue\lg_env\deliverables\mobile_api_draft_20260929')
d=Document(); s=d.sections[0]
s.page_width=Cm(21); s.page_height=Cm(29.7)
s.top_margin=Cm(1.8); s.bottom_margin=Cm(1.8); s.left_margin=Cm(1.8); s.right_margin=Cm(1.8)
for name in ['Normal','Title','Subtitle','Heading 1','Heading 2']:
 st=d.styles[name]; st.font.name='Microsoft JhengHei'; st.font.color.rgb=RGBColor(0,0,0)
 st.element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Microsoft JhengHei')
 st.font.size=Pt(10 if name=='Normal' else {'Title':23,'Subtitle':11,'Heading 1':16,'Heading 2':12}[name])
 st.paragraph_format.space_after=Pt(4)
d.styles['Normal'].paragraph_format.line_spacing=1.05
for n in ['Heading 1','Heading 2']: d.styles[n].paragraph_format.space_before=Pt(10)
d.core_properties.title='傷口高度量測 SDK API 溝通初稿'
d.core_properties.author='算法開發團隊'
def p(t,style=None): return d.add_paragraph(t,style)
def h(t): d.add_heading(t,1)
def table(headers,rows,widths):
 t=d.add_table(rows=1, cols=len(headers)); t.alignment=WD_TABLE_ALIGNMENT.CENTER; t.autofit=False
 for c,w in zip(t.columns,widths): c.width=Cm(w)
 for c,txt in zip(t.rows[0].cells,headers): c.text=txt
 for row in rows:
  for c,txt in zip(t.add_row().cells,row): c.text=txt
 for i,row in enumerate(t.rows):
  trPr=row._tr.get_or_add_trPr(); no=OxmlElement('w:cantSplit'); trPr.append(no)
  if i==0: trPr.append(OxmlElement('w:tblHeader'))
  for c,w in zip(row.cells,widths):
   c.width=Cm(w); c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
   pr=c._tc.get_or_add_tcPr(); sh=OxmlElement('w:shd'); sh.set(qn('w:fill'),'DCE6EF' if i==0 else 'FFFFFF'); pr.append(sh)
   mar=OxmlElement('w:tcMar')
   for side in ['top','left','bottom','right']:
    v=OxmlElement('w:'+side); v.set(qn('w:w'),'65'); v.set(qn('w:type'),'dxa'); mar.append(v)
   pr.append(mar)
   for para in c.paragraphs:
    para.paragraph_format.space_after=Pt(2); para.paragraph_format.space_before=Pt(2)
    for r in para.runs: r.font.size=Pt(9.5); r.bold=(i==0)
 borders=OxmlElement('w:tblBorders')
 for side in ['top','left','bottom','right','insideH','insideV']:
  b=OxmlElement('w:'+side); b.set(qn('w:val'),'single'); b.set(qn('w:sz'),'4'); b.set(qn('w:color'),'D9D9D9'); borders.append(b)
 t._tbl.tblPr.append(borders)
 p('')
def page(): d.add_page_break()
p('傷口高度量測 SDK\nAPI 溝通初稿','Title')
p('版本 v0.1  ｜  2026 年 9 月 29 日  ｜  提供手機 App 開發團隊討論','Subtitle')
p('本文件用於對齊手機 App 與算法 library 的整合邊界、呼叫流程及資料格式。App 負責影像取得與互動顯示，SDK 負責分析影像並回傳相對於指定參考平面的傷口高度。App 不需操作算法內部步驟。')
p('文件狀態：本版為介面提案，API 名稱、平台封裝、輸入限制及效能指標尚待雙方確認；不代表 SDK 已實作或已承諾支援。')
h('1 第一版範圍與分工')
p('建議第一版採錄影完成後分析，再於選定畫面上點選量測。照片序列可使用相同資料入口，但需符合拍攝與標定要求；單張照片或任意相簿影片不預設可量測。即時連續量測、完整深度圖及面積或體積計算另行討論。')
table(['項目','App 團隊','算法 SDK 團隊'],[
('影像輸入','拍攝、選檔、解碼、提供影格與時間','定義拍攝要求及檢查輸入'),
('標定與設定','提供相機資訊並使用約定設定檔','提供設定格式及適用條件'),
('量測操作','顯示量測畫面、轉換點選座標','分析影像、計算高度及判定有效性'),
('結果處理','顯示、儲存、操作提示及重拍流程','回傳數值、狀態、錯誤碼及版本')],[2.6,7.1,7.7])
p('交付方向：C++ 核心搭配平台 wrapper。Android 與 iOS 的交付格式、最低系統版本及優先支援平台，於介面確認時定稿。本文件描述本機 SDK 呼叫，不是 HTTP API。')
h('2 建議操作流程')
p('建立 Session → 提交影像序列 → 啟動分析 → 取得量測畫面 → 設定參考平面 → 點選並取得高度 → 釋放 Session。')
p('同一 Session 可重複量測不同位置。重新分析影像後，舊畫面、平面與結果識別碼不得繼續使用。')
page(); h('3 建議 API 清單')
p('以下為平台無關的邏輯名稱。正式 Kotlin／Swift method signature 於平台封裝確認後提供。sessionId、planeId 等均為 SDK 產生的不透明識別碼。')
table(['API','輸入','輸出與用途'],[
('createSession','SDK 設定、calibrationId','sessionId；建立獨立量測工作'),
('submitFrames','sessionId、FrameInput 清單','接收狀態；提交已解碼影格'),
('analyze','sessionId','jobId；非同步開始分析'),
('getMeasurementView','sessionId','viewId、量測影像、尺寸與來源 frameId'),
('setReferencePlane','sessionId、平面設定','planeId；設定高度零位基準'),
('measureHeight','sessionId、viewId、planeId、點座標','jobId；非同步回傳 MeasurementResult'),
('cancel','sessionId、jobId','取消受理狀態；結果以完成事件為準'),
('releaseSession','sessionId','釋放影像、快取及結果資源')],[4.1,5.5,7.8])
p('setReferencePlane 模式：DEFAULT 使用 SDK 提供的標準基準；CUSTOM_POINTS 使用 App 在量測畫面上選取的參考點。自訂模式建議納入第一版，至少 3 點且不可共線，仍須通過 SDK 有效性檢查。建立平面可能耗時，平台 wrapper 應以背景工作執行。')
h('4 非同步事件與狀態')
table(['事件','必要內容','App 處理'],[
('onProgress','jobId、stage、progress（可省略）','顯示處理中；未知進度不假顯示百分比'),
('onCompleted','jobId、status、對應結果或 error','結束等待，依結果更新畫面')],[4.1,6.0,7.3])
p('Session 狀態：CREATED → INPUT_READY → ANALYZING → READY → CLOSED。分析失敗可保留輸入重試；取消分析後回 INPUT_READY，取消單點量測後仍保持 READY。每個 job 僅有一次終止事件，包含成功、失敗或取消。')
p('同一 Session 第一版建議一次只接受一個工作；忙碌時回 BUSY。回呼可能來自背景執行緒，App 更新 UI 前需切換至主執行緒。releaseSession 完成後不再發出該 Session 的回呼。')
page(); h('5 輸入資料契約')
p('建議由 App 解碼影片，再傳入影格；SDK 不直接依賴 App 的檔案路徑或影片容器。長影片採分批提交或按需取幀的方式，於手機記憶體評估後定稿，不能預設整段影片全部載入 RAM。')
table(['FrameInput 欄位','要求','說明'],[
('frameId','必要','Session 內唯一且有序的影格識別碼'),
('timestampNs','必要','同一序列使用同一時基，單位 ns'),
('cameraId / calibrationId','必要','相機與適用標定版本，禁止混用'),
('imageBuffer','必要','已解碼像素；實際支援格式待確認'),
('width / height / stride','必要','像素尺寸與每列 bytes；格式需明確'),
('pixelFormat','必要','候選 RGB8 或 RGBA8；YUV adapter 另議'),
('rotation / mirrored','必要','方向及鏡像標記，不可只依螢幕方向推定'),
('geometryState','必要','是否已去畸變；建議傳原始影像，由 SDK 處理')],[5.2,2.0,10.2])
p('建議初版 submitFrames 在成功返回前複製需要保留的資料；返回後 App 可重用原 buffer。若後續採零拷貝，須另訂 retain／release 規則，不沿用此假設。')
h('6 相機設定與拍攝條件')
p('calibrationId 應對應相機內參、畸變資訊、解析度與鏡頭設定。校正資料格式由 SDK 團隊提供，App 不需自行推導參數。裁切、數位變焦、鏡頭切換或解析度改變時，不可直接沿用不相容的設定。')
p('現有流程需要符合規格的實體標記及有效視角變化；標記樣式、實際尺寸、拍攝距離、移動方式與可接受影格數，需由 SDK 團隊提供拍攝規格。缺少必要條件時應回傳無法量測，而非保證產生高度。')
h('7 畫面與點選座標')
p('getMeasurementView 回傳 SDK 的量測影像。App 應以此畫面接受點選，座標原點為左上角，x 向右、y 向下，單位為 pixel，可使用小數。有效範圍為 0 ≤ x < width、0 ≤ y < height。')
p('App 必須移除 UI 縮放、留白與裁切造成的座標偏移，再提交量測點。第一版不直接接受原影片畫面或螢幕座標。viewId 用來防止使用過期或不同畫面的點座標。')
page(); h('8 量測結果與高度定義')
p('本介面回傳指定位置相對於參考平面的有號高度，單位固定為 mm；不是相機到傷口的距離。平面法向量由 SDK 固定，建議正值表示朝向量測相機側、負值表示反側；此正負約定需在整合測資中確認。App 不可自行取絕對值。')
table(['MeasurementResult 欄位','型別','說明'],[
('measurementId / jobId','string','量測識別碼與工作識別碼'),
('sessionId / viewId / planeId','string','結果所屬工作、畫面及參考平面'),
('point','{x, y}','SDK 量測畫面上的輸入座標'),
('status','enum','VALID 或 INVALID；判斷數值能否顯示'),
('heightMm','number 或 null','有效量測高度；失敗時必須為 null'),
('referenceType','enum','DEFAULT 或 CUSTOM_POINTS'),
('errorCode / message','string 或 null','程式判斷用代碼及診斷文字'),
('sdkVersion / configVersion','string','重現問題及追蹤設定版本')],[6.1,3.5,7.8])
p('數值 0 表示與參考平面同高，不表示失敗。App 依 status 與 errorCode 決定流程，不解析 message。畫面小數位與最終顯示字詞由雙方確認，不能把顯示位數當成量測精度。')
h('9 呼叫示例')
p('以下 JSON 僅示意 wrapper 資料結構，不代表需要網路傳輸；數值為介面範例。')
for line in ['measureHeight({ sessionId: "s01", viewId: "v01",', '  planeId: "p01", point: { x: 640.5, y: 360.0 } })', '→ { jobId: "j02" }', '', 'onCompleted({ jobId: "j02", status: "SUCCEEDED",', '  result: { measurementId: "m01", sessionId: "s01",', '    viewId: "v01", planeId: "p01",', '    point: { x: 640.5, y: 360.0 }, status: "VALID",', '    heightMm: -1.25, referenceType: "DEFAULT",', '    errorCode: null, message: null,', '    sdkVersion: "待定", configVersion: "待定" } })']:
 q=p(line); q.paragraph_format.space_after=Pt(0)
 for r in q.runs: r.font.name='Consolas'; r.font.size=Pt(9)
p('量測無效時，工作仍可正常完成並回傳 status=INVALID、heightMm=null 與原因；系統或資源錯誤則以工作 FAILED 回覆。')
page(); h('10 錯誤處理與整合驗收')
table(['代碼','意義','建議 App 行為'],[
('INVALID_ARGUMENT','欄位或點座標錯誤','修正呼叫，不自動重拍'),
('INVALID_STATE / BUSY','呼叫時機錯誤或工作中','等待工作完成或調整流程'),
('CALIBRATION_MISMATCH','影像與相機設定不相容','載入正確設定後重新分析'),
('INSUFFICIENT_INPUT','影格或必要資訊不足','提示補充資料或重拍'),
('ANALYSIS_FAILED','影像無法建立量測條件','依原因提示改善拍攝條件'),
('PLANE_UNAVAILABLE','無法建立參考平面','重選參考點或重新分析'),
('MEASUREMENT_INVALID','該位置無有效量測','顯示無法量測，允許重新點選'),
('STALE_HANDLE','Session 或畫面識別碼失效','取得新畫面與平面再量測'),
('CANCELLED','工作已取消','結束等待，不顯示錯誤警告'),
('RESOURCE_EXHAUSTED / INTERNAL_ERROR','資源不足或內部錯誤','釋放資源並記錄診斷資訊')],[6.5,4.8,6.1])
p('雙方共同建立固定測資與預期結果。SDK 團隊提供測試影像、適用設定及允許誤差；App 團隊驗證方向與座標映射、有效與無效結果、取消重試、重複量測及資源釋放。')
p('效能驗收以指定機型量測分析時間、單點回應時間、峰值記憶體及套件大小。目標值待確認，初稿不承諾精度、延遲或 FPS。診斷至少保留版本、工作 ID 與錯誤碼；原始影像的保存與傳遞方式由雙方另訂。')
h('11 首次對接需確認事項')
for t in [
'平台與交付：Android／iOS 優先順序、App 技術棧、最低系統版本、目標機型及 SDK 封裝格式。',
'輸入方式：錄影後分析或照片序列、影片解碼責任、像素格式、解析度、長度上限及取幀方式。',
'拍攝與設定：標記規格、相機標定供應方式、是否允許相簿匯入及鏡頭切換。',
'量測互動：第一版是否包含自訂平面、正負值顯示方式，以及點選座標轉換責任。',
'交付與驗收：測資、範例 App、API 版本策略、錯誤碼清單、效能與精度目標。']:
 p('• '+t)
p('確認上述項目後，SDK 團隊提供正式 API header、平台 wrapper 範例、資料格式及整合測試清單，作為開發與驗收依據。')
footer=s.footer.paragraphs[0]; footer.alignment=2
r=footer.add_run('API 溝通初稿 v0.1  |  '); r.font.size=Pt(8)
f=OxmlElement('w:fldSimple'); f.set(qn('w:instr'),'PAGE'); footer._p.append(f)
path=out/'Wound_Height_SDK_API_Draft_v0.1.docx'
# Previous version preserved in revision_backup_01
d.save(path)
print(path)
print('tables',len(d.tables),'paragraphs',len(d.paragraphs))

