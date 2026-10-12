"""Build an anonymous PDF design report from Markdown and recorded evidence.

Optional document tool dependency: reportlab. Windows uses installed SimSun;
other systems can pass an embeddable TrueType Chinese font with --font.
No experiments are started and no runtime records are modified.
--freeze-evidence rebuilds the bundled historical acceptance section only; its
fixed environment/incident narrative is not a template for current experiments.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import statistics
from pathlib import Path

from dgfl.experiments.evidence import validate_record, validate_run_id, validate_suite, validate_suite_hash

ROOT=Path(__file__).resolve().parents[1]
# All formal/full-data/fault records cited by the fixed acceptance narrative were
# produced by this source snapshot. Do not relabel newer runs with that narrative.
HISTORICAL_REPORT_SOURCE='ac00368af00ca5356430ef748e849cbb424b3341756cabba664edbb461fa95ad'


def load_suite(folder):
    state=json.loads((folder/'suite.json').read_text('utf8'))
    suite=validate_suite(json.loads((folder/'config.json').read_text('utf8')),normalize=False)
    validate_suite_hash(suite,state.get('suite_hash'))
    mapping=state['cases']; declared=suite['cases']
    if set(mapping)!={x['name'] for x in declared}: raise ValueError('suite cases incomplete')
    ids=[validate_run_id(value) for value in mapping.values()]
    if len(ids)!=len(set(ids)): raise ValueError('duplicate mapped run identifier')
    resolved=state.get('resolved_configs',{})
    if not isinstance(resolved,dict) or not set(resolved)<=set(mapping):
        raise ValueError('resolved configurations contain undeclared mappings')
    result={}
    for case in declared:
        record=json.loads((folder/mapping[case['name']]/'result.json').read_text('utf8'))
        validate_record(record,case['config'],mapping[case['name']],resolved_config=resolved.get(case['name']))
        if record['status'] not in ('completed','aborted','failed'): raise ValueError('suite is still running')
        result[case['name']]=record
    return result


def percentage(value): return '无已发布模型' if value is None else f'{100*value:.2f}%'


def comparable(a,b):
    # Different mode and requested number of rounds are intentional; each compared
    # round must still have identical data, attack, optimization inputs and result set.
    ca={k:v for k,v in a['config'].items() if k not in ('mode','rounds')}
    cb={k:v for k,v in b['config'].items() if k not in ('mode','rounds')}
    if ca!=cb: raise ValueError('unmatched experiment configuration')
    if a['evidence']['implementation']['source_sha256']!=b['evidence']['implementation']['source_sha256']:
        raise ValueError('unmatched running source version')
    pa={k:v['partition_hash'] for k,v in a['evidence']['partitions'].items()}
    pb={k:v['partition_hash'] for k,v in b['evidence']['partitions'].items()}
    if pa!=pb: raise ValueError('unmatched data partitions')


def evidence_section(folder):
    formal=load_suite(folder/'formal'); full=load_suite(folder/'full-data')
    for record in (*formal.values(),*full.values()):
        if record.get('evidence',{}).get('implementation',{}).get('source_sha256')!=HISTORICAL_REPORT_SOURCE:
            raise ValueError('unmatched historical report source; current experiments need a separately reviewed report')
    lines=['以下为旧逐坐标证明、固定二人批次及三云拓扑的冻结历史验收，不是当前 LegoGroth16 版本的性能。'
           '历史实测主机为 Intel Core i7-14700HX（20 核、28 线程）、约 16 GB 内存，Windows 11。'
           '当时使用单机 12 角色进程，训练及密码运算均为 CPU。存在后台活动及进程缓存，采用固定顺序，'
           '以下为该环境的描述性测量，不是独立重复试验的置信区间。','',
           '| 种子 | A/C/D 首轮模型一致 | 首轮准确率 | C 轮时/s | D 轮时/s | C / D 应用层 MiB |',
           '|---|---|---|---|---|---|']
    ctimes=[]; dtimes=[]; cbytes=[]; dbytes=[]
    for seed in (42,43,44):
        a=formal[f'plain_clean_{seed}']; c=formal[f'dgflow_clean_{seed}']; d=formal[f'optimized_clean_{seed}']
        comparable(a,c); comparable(c,d)
        if not all(x['rounds'] for x in (a,c,d)):
            lines.append(f'| {seed} | 存在未完成首轮，见原始记录 | 不作成功比较 | — | — | — |'); continue
        ar,cr,dr=(x['rounds'][0] for x in (a,c,d))
        equal=ar['model_hash']==cr['model_hash']==dr['model_hash'] and ar['accepted_clients']==cr['accepted_clients']==dr['accepted_clients']
        if not equal or not ar['round']==cr['round']==dr['round']==1:
            raise ValueError(f'unmatched first-round results for seed {seed}; no performance comparison published')
        ctimes.append(cr['duration_s']); dtimes.append(dr['duration_s']); cbytes.append(cr['bytes_sent']); dbytes.append(dr['bytes_sent'])
        lines.append(f'| {seed} | {"是" if equal else "否"} | {percentage(dr["accuracy"])} | {cr["duration_s"]:.2f} | {dr["duration_s"]:.2f} | {cr["bytes_sent"]/2**20:.2f} / {dr["bytes_sent"]/2**20:.2f} |')
    if ctimes:
        lines+=['',f'匹配首轮的 C 平均轮时为 {statistics.mean(ctimes):.2f} 秒，D 为 {statistics.mean(dtimes):.2f} 秒；'
                 f'平均轮时比 C/D 为 {statistics.mean(ctimes)/statistics.mean(dtimes):.2f}。'
                 f'D 应用层字节数相对 C 的均值变化为 {(statistics.mean(dbytes)/statistics.mean(cbytes)-1)*100:+.2f}%。'
                 '该比较保留全部匹配种子；42 的 C 仅运行一轮，因此没有用 C 单轮总时长与 D 三轮总时长比较。']
    a=formal['plain_clean_42']; d=formal['optimized_clean_42']; comparable(a,d)
    equal=len(a['rounds'])==len(d['rounds'])==3 and all(x['model_hash']==y['model_hash'] and x['accepted_clients']==y['accepted_clients'] for x,y in zip(a['rounds'],d['rounds']))
    lines+=['',f'种子 42 的 A/D 三轮整数模型摘要与批准集合{"全部相同" if equal else "未全部相同，须查看记录"}。'
              'D 每轮准确率为 '+'、'.join(percentage(r['accuracy']) for r in d['rounds'])+'。',
            '', '| 方向反转种子 | 模式 | 最后已发布模型准确率 | 完成轮中的攻击者获准/提交 | 完成轮中的诚实退出/人数 | 状态 |',
            '|---|---|---|---|---|---|']
    for seed in (42,43,44):
        for mode in ('encrypted','optimized'):
            r=formal[f'{mode}_sign_flip_{seed}']; rows=r['rounds']
            lines.append(f'| {seed} | {mode} | {percentage(r["summary"].get("accuracy"))} | '
                         f'{sum(x["attack_accepted"] for x in rows)}/{sum(x["attack_submitted"] for x in rows)} | '
                         f'{sum(x["honest_rejected"] for x in rows)}/{sum(x["honest_total"] for x in rows)} | {r["status"]} |')
    lines+=['','诚实退出包含固定批次连带退出，不能全部称作相似度检测误报。上述计数只覆盖已发布模型的轮次；'
            '未完成轮的验证记录单独保留，不以 0/0 推断没有拒绝或没有攻击。',
            '', '方向反转 B 模式种子43运行中发生系统自动休眠，Windows 系统日志记录的 UTC 区间为 '
            '10:01:59 至 12:18:17。原始墙钟含休眠时间，未扣除或重写；该例仅用于正确性及攻击结果比较，'
            '不用于性能加速结论。事件来源见 environment-events.json。',
            '', '| 单轮补充场景（种子42） | 准确率 | 获准客户端 | 直接证明失败 | 连带退出 | 状态 |',
            '|---|---|---|---|---|---|']
    for name,label in [('optimized_tamper_42','篡改公开范数'),('optimized_non_iid_42','诚实非IID'),('optimized_label_flip_42','标签翻转')]:
        r=formal[name]; row=r['rounds'][-1] if r['rounds'] else {}; vals=row.get('validations',r.get('current_validations',[]))
        lines.append(f'| {label} | {percentage(r["summary"].get("accuracy"))} | {", ".join(row.get("accepted_clients",[])) or "无模型发布"} | '
                     f'{sum(v.get("proof_valid") is False for v in vals)} | {", ".join(row.get("collateral_clients",[])) or "无/未形成结果"} | {r["status"]} |')
    lines+=['','这些补充场景各只有一个种子，不支持对所有非IID、标签投毒或后门攻击的普遍防御结论。'
            '未被识别的有效证明投毒也如实保留；证明正确不等于训练行为正确。']
    fa=full['plain_full_mnist_42']; fd=full['optimized_full_mnist_42']; comparable(fa,fd)
    if [r['round'] for r in fa['rounds']]!=[r['round'] for r in fd['rounds']]:
        raise ValueError('unmatched full-data rounds; refusing a truncated comparison')
    lines+=['','完整 MNIST 实验使用官方 60,000 训练/10,000 测试样本，均匀分给六客户端，种子42。',
            '', '| 轮次 | 全数据明文准确率 | 全数据加密准确率 | 整数模型及批准集合一致 | 加密轮时/s |',
            '|---|---|---|---|---|']
    for a,d in zip(fa['rounds'],fd['rounds']):
        eq=a['model_hash']==d['model_hash'] and a['accepted_clients']==d['accepted_clients']
        lines.append(f'| {d["round"]} | {percentage(a["accuracy"])} | {percentage(d["accuracy"])} | {"是" if eq else "否"} | {d["duration_s"]:.2f} |')
    records=list(formal.values())+list(full.values()); statuses={s:sum(r['status']==s for r in records) for s in ('completed','aborted','failed')}
    lines+=['',f'上述预声明套件共 {len(records)} 个任务：completed={statuses["completed"]}，aborted={statuses["aborted"]}，failed={statuses["failed"]}。'
            '逐例配置及运行编号见 evidence 下的 suite.json 和 result.json。',
            '', '| 真实进程故障场景 | 实际完成轮数 | 终态 | 结果说明 |','|---|---|---|---|']
    faults=json.loads((folder/'faults'/'index.json').read_text('utf8'))
    if faults.get('status')!='passed' or faults.get('restoration',{}).get('status')!='restored':
        raise ValueError('fault suite or node restoration did not pass')
    fault_labels={'one_aggregator_down':'停止一个聚合节点','two_aggregators_down':'停止两个聚合节点','one_client_down':'停止一个客户端'}
    for item in faults['cases']:
        r=json.loads((folder/'faults'/item['run_id']/'result.json').read_text('utf8'))
        if r.get('evidence',{}).get('implementation',{}).get('source_sha256')!=HISTORICAL_REPORT_SOURCE:
            raise ValueError('unmatched historical report source in fault evidence')
        lines.append(f'| {fault_labels[item["name"]]} | {len(r["rounds"])} | {r["status"]} | {item["observation"]} |')
    tests=json.loads((folder/'test-summary.json').read_text('utf8'))
    lines+=['',f'该历史版本的自动化验收：{tests["passed"]} 项通过，{tests["skipped"]} 项跳过，{tests["failed"]} 项失败。'
            +tests.get('skip_explanation',''),
            '', '正式性能运行前统一重启全部本机角色，运行中冻结协议及训练源码。'
            '随后修复证据导出的恢复与未完成轮统计，以及未返回攻击客户端的提交数口径；'
            '这些修正不改变密码计算或模型训练。原始记录内源码指纹未被覆盖，逐文件版本差异及历史计数核对见 release-provenance.json。']
    return '\n'.join(lines)


def render_pdf(source,output,font):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        CondPageBreak,
        Image,
        KeepTogether,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    pdfmetrics.registerFont(TTFont('CN',str(font),subfontIndex=0))
    pdfmetrics.registerFontFamily('CN',normal='CN',bold='CN',italic='CN',boldItalic='CN')
    title=ParagraphStyle('title',fontName='CN',fontSize=22,leading=33,spaceAfter=24,textColor=colors.HexColor('#143c43'),alignment=TA_CENTER)
    body=ParagraphStyle('body',fontName='CN',fontSize=10.4,leading=17,spaceAfter=8,wordWrap='CJK',allowWidows=0,allowOrphans=0)
    h1=ParagraphStyle('h1',parent=body,fontSize=15,leading=22,spaceBefore=17,spaceAfter=9,textColor=colors.HexColor('#156b72'),keepWithNext=True)
    h2=ParagraphStyle('h2',parent=body,fontSize=12,leading=19,spaceBefore=10,spaceAfter=7,textColor=colors.HexColor('#143c43'),keepWithNext=True)
    table_style=ParagraphStyle('cell',parent=body,fontSize=8.3,leading=12,spaceAfter=0)
    caption=ParagraphStyle('caption',parent=body,fontSize=8.5,leading=13,alignment=TA_CENTER,textColor=colors.HexColor('#52666b'))
    def inline(text):
        value=html.escape(text)
        value=re.sub(r'`([^`]+)`',r'<font color="#18575e">\1</font>',value)
        value=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',value)
        return value
    lines=source.read_text('utf8').splitlines(); story=[]; i=0; width=A4[0]-104
    while i<len(lines):
        line=lines[i].strip(); i+=1
        if not line or line.startswith('<!--'): continue
        if line.startswith('|'):
            raw=[line]
            while i<len(lines) and lines[i].strip().startswith('|'): raw.append(lines[i].strip()); i+=1
            rows=[[x.strip() for x in row.strip('|').split('|')] for row in raw if not re.fullmatch(r'[|:\-\s]+',row)]
            n=len(rows[0]); widths=[width/n]*n
            if n==4: widths=[width*.19,width*.22,width*.26,width*.33]
            data=[[Paragraph(inline(x),table_style) for x in row] for row in rows]
            table=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e4f0ef')),
                ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
                ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
                ('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#64a6a8')),
                ('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#d3dfdf'))]))
            story.extend([table,Spacer(1,12)]); continue
        match=re.fullmatch(r'!\[(.*?)\]\((.*?)\)',line)
        if match:
            path=source.parent/match[2]
            if not path.exists(): raise ValueError(f'missing report image: {path.name}')
            image=Image(str(path)); ratio=width/image.imageWidth
            image.drawWidth=width; image.drawHeight=image.imageHeight*ratio
            story.append(KeepTogether([image,Spacer(1,5),Paragraph(inline(match[1]),caption),Spacer(1,10)])); continue
        if line.startswith('# '):
            text=inline(line[2:]).replace('隐私保护','<br/>隐私保护',1)
            story.append(Paragraph(text,title)); continue
        if line.startswith('## '):
            following=next((x.strip() for x in lines[i:] if x.strip()),'')
            figure=re.fullmatch(r'!\[(.*?)\]\((.*?)\)',following)
            if figure:
                item=Image(str(source.parent/figure[2]))
                story.append(CondPageBreak(item.imageHeight*width/item.imageWidth+85))
            story.append(Paragraph(inline(line[3:]),h1)); continue
        if line.startswith('### '): story.append(Paragraph(inline(line[4:]),h2)); continue
        if re.match(r'^\d+\.\s',line): story.append(Paragraph(inline(line),body)); continue
        paragraph=[line]
        while i<len(lines) and lines[i].strip() and not lines[i].lstrip().startswith(('#','|','![','<!--')) and not re.match(r'^\d+\.\s',lines[i].strip()):
            paragraph.append(lines[i].strip()); i+=1
        story.append(Paragraph(inline(' '.join(paragraph)),body))
    def footer(canvas,doc):
        canvas.setFont('CN',8); canvas.setFillColor(colors.HexColor('#6c7e81'))
        canvas.drawRightString(A4[0]-52,27,f'{doc.page}')
    output.parent.mkdir(parents=True,exist_ok=True)
    document=SimpleDocTemplate(str(output),pagesize=A4,leftMargin=52,rightMargin=52,topMargin=46,bottomMargin=46,
                               title='基于去中心化函数加密的隐私保护鲁棒联邦学习系统',author='',subject='匿名作品设计报告')
    document.build(story,onFirstPage=footer,onLaterPages=footer)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'docs/submission/design-report.md')
    parser.add_argument('--output',type=Path,default=ROOT/'dist/submission/design-report.pdf')
    parser.add_argument('--font',type=Path,default=Path(os.environ.get('WINDIR','/'))/'Fonts'/'simsun.ttc')
    parser.add_argument('--freeze-evidence',action='store_true')
    args=parser.parse_args()
    if args.freeze_evidence:
        section=evidence_section(args.source.parent/'evidence')
        text=args.source.read_text('utf8')
        text=re.sub(r'<!-- BEGIN VERIFIED RESULTS -->.*?<!-- END VERIFIED RESULTS -->',
                    '<!-- BEGIN VERIFIED RESULTS -->\n'+section+'\n<!-- END VERIFIED RESULTS -->',text,flags=re.S)
        args.source.write_text(text,encoding='utf8')
    render_pdf(args.source,args.output,args.font)
    print(args.output)


if __name__=='__main__': main()
