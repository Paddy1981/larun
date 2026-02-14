'use client'

/**
 * File Upload Component
 *
 * Drag-and-drop interface for FITS file uploads
 */

import { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, File, X } from 'lucide-react'

interface FileUploadProps {
  onFileSelect: (file: File | null) => void
  selectedFile: File | null
  disabled?: boolean
}

export function FileUpload({ onFileSelect, selectedFile, disabled }: FileUploadProps) {
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      onFileSelect(acceptedFiles[0])
    }
  }, [onFileSelect])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/fits': ['.fits', '.fit'],
      'application/octet-stream': ['.fits', '.fit'],
    },
    multiple: false,
    disabled,
  })

  const handleRemove = (e: React.MouseEvent) => {
    e.stopPropagation()
    onFileSelect(null)
  }

  return (
    <div>
      {!selectedFile ? (
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-lg p-12 text-center transition-colors cursor-pointer ${
            isDragActive
              ? 'border-larun-black bg-larun-lighter-gray'
              : 'border-larun-light-gray hover:border-larun-medium-gray'
          } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          <input {...getInputProps()} />
          <Upload className="w-12 h-12 text-larun-medium-gray mx-auto mb-4" />
          <p className="text-larun-black font-medium mb-2">
            {isDragActive ? 'Drop FITS file here' : 'Drag & drop FITS file'}
          </p>
          <p className="text-sm text-larun-medium-gray mb-4">
            or click to browse
          </p>
          <p className="text-xs text-larun-medium-gray">
            Supports: .fits, .fit (max 50 MB)
          </p>
        </div>
      ) : (
        <div className="border-2 border-larun-black rounded-lg p-6 bg-larun-lighter-gray">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded bg-larun-black text-larun-white flex items-center justify-center">
                <File className="w-5 h-5" />
              </div>
              <div>
                <p className="font-medium text-larun-black">{selectedFile.name}</p>
                <p className="text-sm text-larun-medium-gray">
                  {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
            </div>
            {!disabled && (
              <button
                onClick={handleRemove}
                className="p-2 hover:bg-larun-white rounded transition-colors"
              >
                <X className="w-5 h-5 text-larun-medium-gray" />
              </button>
            )}
          </div>
        </div>
      )}

      <p className="text-xs text-larun-medium-gray mt-3">
        Supported sources: TESS, Kepler, K2, or your own observations
      </p>
    </div>
  )
}
