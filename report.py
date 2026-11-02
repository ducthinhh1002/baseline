"""20-row report matching the requested Method / Model / seven attacks layout."""
import csv,json
from common import ROOT,config,read,write,METHODS,STUDENTS
def main():
    c=config();rows=[];details=[]
    for method in c['methods']:
        for model in ['teacher']+c['students']:
            cells={'Method':METHODS[method],'Model':'Teacher' if model=='teacher' else f'student {STUDENTS[model]} LoRA rank4'}
            for attack in c['attacks']:
                p=ROOT/'teachers'/method/'results'/f'{attack}.json' if model=='teacher' else ROOT/'students'/method/model/attack/'results.json'
                value=None
                if p.exists():
                    a=read(p);value=next((r['acceptance'] for r in a['rows'] if r['images_per_group']==256),None)
                    group={r['images_per_group']:r for r in a['rows']}
                    details.append({'Method':METHODS[method],'Model':model,'Training manipulation':attack,
                        'num_bits':a.get('num_bits'),'min_correct':a.get('acceptance_min_correct'),
                        'theoretical_bit_fpr':a.get('theoretical_fpr'),
                        'nominal_fpr':a.get('nominal_fpr'),
                        'BA/image':group.get(1,{}).get('bit_accuracy'),
                        'NK/image':group.get(1,{}).get('acceptance'),
                        'NK agg8':group.get(8,{}).get('acceptance'),
                        'NK agg32':group.get(32,{}).get('acceptance'),
                        'NK agg256':group.get(256,{}).get('acceptance'),
                        'BA agg256':group.get(256,{}).get('bit_accuracy'),
                        'metric_kind':'native bit acceptance' if a.get('num_bits') else 'calibrated scalar detection'})
                cells[attack]='pending' if value is None else f'{value*100:.2f}%'
            rows.append(cells)
    out=ROOT/'reports';out.mkdir(exist_ok=True)
    with (out/'acceptance_agg256.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(out/'table.json',{'metric':'acceptance agg256; native payload lengths; SERUM (and optional GaussMarker fused mode) use clean-calibrated scalar thresholds','rows':rows})
    if details:
        with (out/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(details[0]));writer.writeheader();writer.writerows(details)
    write(out/'metrics.json',{'values':'fractions for BA/NK; no FID/KID',
        'missing':'null means inapplicable or not evaluated; never zero-filled','rows':details})
    return out
if __name__=='__main__':print(main())
