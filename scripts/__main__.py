import click

@click.command()
@click.argument('file_path', required=True, type=click.Path(exists=True, dir_okay=True))
@click.argument('dst_dir', required=True, type=click.Path(exists=True, dir_okay=True))
@click.option('--aoi_geojson', '-aoi', required=False, type=str, help="Geojson file path if you want to mask sentinel 2 data.")
def main(file_path, dst_dir, aoi_geojson):
    from sen4biophysical import biophysical

    biophysical(file_path, dst_dir, aoi_geojson)

if __name__ == '__main__':
    main()