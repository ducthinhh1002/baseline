"""Optional utility reporting; no watermark decision, no import-time downloads."""
def image_quality(real_dir,generated_dir,prompts,clip_model_path):
    import torch_fidelity
    from transformers import CLIPModel,CLIPProcessor
    from PIL import Image
    import torch
    from pathlib import Path
    result=torch_fidelity.calculate_metrics(input1=str(real_dir),input2=str(generated_dir),cuda=True,
            fid=True,kid=True,kid_subset_size=min(1000,len(prompts)),verbose=False)
    model=CLIPModel.from_pretrained(clip_model_path,local_files_only=True).cuda().eval()
    processor=CLIPProcessor.from_pretrained(clip_model_path,local_files_only=True)
    scores=[]
    for i,row in enumerate(prompts):
        image=Image.open(Path(generated_dir)/f'sample_{i:05d}.png').convert('RGB')
        data=processor(text=[row['text']],images=[image],return_tensors='pt',padding=True,truncation=True).to('cuda')
        with torch.no_grad():out=model(**data);scores.append(float(torch.nn.functional.cosine_similarity(out.image_embeds,out.text_embeds)))
    result['clip_cosine_similarity']=sum(scores)/len(scores)
    result['clip_definition']='raw CLIP cosine; not silently rescaled to CLIPScore'
    return result
