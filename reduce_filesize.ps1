# Define the directory containing TIF files
$directoryPath = "photos"

# Define the maximum file size in bytes (80 MB)
$maxFileSize = 80 * 1MB

# Function to get the file size
function Get-FileSize {
    param (
        [string]$filePath
    )
    $fileInfo = Get-Item $filePath
    return $fileInfo.Length
}

# Function to resize the image by adjusting resolution
function Resize-Image {
    param (
        [string]$inputFile,
        [string]$outputFile,
        [int]$targetSize
    )
    # Initial guess for the scale factor
    $scaleFactor = 1.0

    do {
        # Calculate new dimensions based on scale factor
        $dimensions = & magick convert $inputFile -ping -format "%[fx:w*$scaleFactor]x%[fx:h*$scaleFactor]" info:
        
        # Resize the image to the new dimensions
        & magick convert $inputFile -resize $dimensions $outputFile

        # Get the new file size
        $fileSize = Get-FileSize $outputFile

        # Adjust the scale factor
        if ($fileSize -gt $targetSize) {
            $scaleFactor *= 0.9  # Reduce scale factor by 10%
        } else {
            $scaleFactor *= 1.1  # Increase scale factor by 10% to get closer to target size
        }

    } while ($fileSize -gt $targetSize -and $scaleFactor -gt 0.1)
}

# Process each TIF file in the directory
Get-ChildItem -Path $directoryPath -Filter "*.tif" | ForEach-Object {
    $filePath = $_.FullName
    $fileSize = Get-FileSize $filePath

    if ($fileSize -gt $maxFileSize) {
        $outputFilePath = [System.IO.Path]::Combine($directoryPath, [System.IO.Path]::GetFileNameWithoutExtension($_.Name) + "_resized.tif")
        Resize-Image -inputFile $filePath -outputFile $outputFilePath -targetSize $maxFileSize
        Write-Output "Resized $filePath to $outputFilePath"
    } else {
        Write-Output "$filePath is already under the size limit."
    }
}