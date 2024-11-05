# Define the directory to search for TIFF files
$directory = "photos"

# Get all TIFF files in the directory
$tiffFiles = Get-ChildItem -Path $directory -Filter *.tif

# Create a variable to store the CSV output
$output = ""

# Loop through each TIFF file
foreach ($file in $tiffFiles) {
    try {
        # Use ImageMagick's identify command to get the dimensions
        $identifyOutput = & magick identify -format "%w,%h" $file.FullName
        
        # Split the output into width and height
        $dimensions = $identifyOutput -split ","
        $width = $dimensions[0]
        $height = $dimensions[1]
        
        # Add the file name and dimensions to the output variable
        $output += "$($file.Name), $width x $height`n"
    } catch {
        Write-Host "Error processing file $($file.FullName): $_"
    }
}

# Write the output to a CSV file
$output | Out-File -FilePath "$directory\TIFFDimensions.csv" -Encoding UTF8

Write-Host "CSV file created at $directory\TIFFDimensions.csv"
cat $directory\TIFFDimensions.csv