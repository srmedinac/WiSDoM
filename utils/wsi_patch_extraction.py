from histoprep import SlideReader

level = 1
overlap = 0
patch_size = 1024
max_background = 0.8
output_path = '/path/to/output'
path = 'path/to/your/slide.tif'  

reader = SlideReader(path)
_, tissue_mask = reader.get_tissue_mask(level=level)
tile_coordinates = reader.get_tile_coordinates(
            tissue_mask=tissue_mask,
            width=patch_size,
            overlap=overlap,
            max_background=max_background,
        )
_ = reader.save_regions(parent_dir=output_path, coordinates=tile_coordinates, level=level, overwrite=True, save_metrics=False, num_workers=12)