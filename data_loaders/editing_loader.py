class ImageEditingDataset:
    def __init__(
        self,
        json_path: str,
        dataset_root: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int,
        resolution: int = 256,
    ):
        with open(json_path) as f:
            self.data = json.load(f)
        self.dataset_root = dataset_root
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.resolution = resolution
        
        # Build sample list
        self.samples = []
        for post_id, post in self.data.items():
            source_path = os.path.join(dataset_root, "dataset", post_id, post["source_name"])
            if not os.path.exists(source_path):
                continue
                
            # Collect all valid edited images
            edited_paths = []
            for comment in post["comments"].values():
                edited_paths.extend([
                    path for path in comment.get("shopped_images", [])
                    if isinstance(path, str) and os.path.exists(path)
                ])
            
            # Create samples
            text = f"{post['title']} {post['body']}".strip()
            for edited_path in edited_paths:
                self.samples.append({
                    "source": source_path,
                    "edited": edited_path,
                    "text": text
                })

        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load and transform images
        source_img = Image.open(sample["source"]).convert("RGB")
        edited_img = Image.open(sample["edited"]).convert("RGB")
        
        return {
            "source_images": self.transform(source_img),
            "edited_images": self.transform(edited_img),
            "input_ids": self.tokenizer(
                sample["text"],
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids[0]
        }