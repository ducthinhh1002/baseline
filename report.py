"""20-row report matching the requested Method / Model / seven attacks layout."""
import csv,json
from common import ROOT,config,read,write,METHODS,STUDENTS
def main():
    c=config();rows=[]
    for method in c['methods']:
        for model in ['teacher']+c['students']:
            cells={'Method':METHODS[method],'Model':'Teacher' if model=='teacher' else f'student {STUDENTS[model]} LoRA rank4'}
            for attack in c['attacks']:
                p=ROOT/'teachers'/method/'results'/f'{attack}.json' if model=='teacher' else ROOT/'students'/method/model/attack/'results.json'
                value=None
                if p.exists():
                    a=read(p);value=next((r['acceptance'] for r in a['rows'] if r['images_per_group']==256),None)
                cells[attack]='pending' if value is None else f'{value*100:.2f}%'
            rows.append(cells)
    out=ROOT/'reports';out.mkdir(exist_ok=True)
    with (out/'acceptance_agg256.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(out/'table.json',{'metric':'acceptance agg256; method-specific payload length; SERUM/GaussMarker fused score uses frozen scalar thresholds','rows':rows})
    return out
if __name__=='__main__':print(main())
